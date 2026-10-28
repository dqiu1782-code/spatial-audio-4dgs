"""HRTF-based real-time binaural renderer for moving sources and a moving listener.

Per source: direct path + 6 first-order wall reflections (image sources), each with
propagation delay (Doppler), 1/r gain and a KEMAR HRIR for its arrival direction;
plus a diffuse late-reverb tail precomputed offline. HRIR switches are cross-faded per block.

HRIRs: MIT KEMAR (Gardner & Martin, 1994), "compact"/"diffuse" sets (symmetric, 44.1 kHz).
"""
import glob
import math
import os
import re
import threading
import wave

import numpy as np

C_SOUND = 343.0


# ---------------------------------------------------------------- HRIR set
def _read_kemar_file(path):
    """Returns (taps, 2) float array; supports .wav and big-endian interleaved .dat."""
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            ch, n = w.getnchannels(), w.getnframes()
            x = np.frombuffer(w.readframes(n), "<i2").astype(np.float64).reshape(-1, ch)
        return x / 32768.0
    x = np.fromfile(path, ">i2").astype(np.float64).reshape(-1, 2)
    return x / 32768.0


class HRTFSet:
    """Symmetric KEMAR set: file H<el>e<az>a holds (left=L(az), right=L(360-az)) for az in [0, 180]."""

    def __init__(self, root, fs):
        from scipy.signal import resample_poly
        files = [p for p in glob.glob(os.path.join(root, "**", "H*e*a.*"), recursive=True)
                 if p.lower().endswith((".wav", ".dat"))]
        if not files:
            raise FileNotFoundError(f"no KEMAR H*e*a files under {root}")
        dirs, irs = [], []
        g = math.gcd(int(fs), 44100)
        for p in files:
            m = re.match(r"H(-?\d+)e(\d+)a", os.path.basename(p))
            el, az = int(m.group(1)), int(m.group(2))
            h = _read_kemar_file(p)
            h = resample_poly(h, int(fs) // g, 44100 // g, axis=0) if fs != 44100 else h
            for a, hh in ((az, h), (360 - az, h[:, ::-1])):     # mirror to the left hemisphere
                if a in (0, 180, 360) and hh is not h:
                    continue
                dirs.append(self.unit(el, a % 360))
                irs.append(hh)
        n = max(len(h) for h in irs)
        self.ir = np.stack([np.pad(h, ((0, n - len(h)), (0, 0))) for h in irs])   # (P, taps, 2)
        self.dir = np.array(dirs)                                                 # (P, 3) head frame
        self.taps = n
        # Normalize so a frontal source has unit RMS gain across ears.
        k = int(np.argmax(self.dir @ np.array([1.0, 0.0, 0.0])))
        self.ir /= np.sqrt(np.sum(self.ir[k] ** 2) / 2)

    @staticmethod
    def unit(el, az):
        """Head frame: x forward, y right, z up; az clockwise (90 = right)."""
        e, a = math.radians(el), math.radians(az)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def nearest(self, v_head):
        return int(np.argmax(self.dir @ (v_head / (np.linalg.norm(v_head) + 1e-12))))


# ---------------------------------------------------------------- room
def image_sources(S, room_min, room_max):
    """First-order image sources of S (..., 3) in an axis-aligned box. Returns (6, ..., 3)."""
    out = []
    for ax in range(3):
        for wall in (room_min[ax], room_max[ax]):
            I = np.array(S, float, copy=True)
            I[..., ax] = 2 * wall - I[..., ax]
            out.append(I)
    return np.stack(out)


def late_reverb_ir(fs, rt60=0.45, predelay=0.025, length=None, seed=0):
    """Stereo decorrelated exponentially decaying noise tail."""
    length = length or rt60 * 1.2
    n = int(length * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    env = np.exp(-6.91 * t / rt60) * (t >= predelay)
    ir = rng.standard_normal((n, 2)) * env[:, None]
    # Gentle high-frequency damping.
    from scipy.signal import lfilter
    ir = lfilter([0.35], [1, -0.65], ir, axis=0)
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


# ---------------------------------------------------------------- renderer
class Source:
    def __init__(self, dry, traj_t, traj_p, name=""):
        self.dry = np.asarray(dry, np.float64)
        self.name = name
        self.set_traj(traj_t, traj_p)

    def set_traj(self, t, p):
        self.tk, self.pk = np.asarray(t, float), np.asarray(p, float)

    def pos(self, ts):
        return np.stack([np.interp(ts, self.tk, self.pk[:, k]) for k in range(3)], -1)


class HRTFRenderer:
    """Streaming binaural renderer; block_render(n) is called from the audio callback."""

    def __init__(self, sources, fs, duration, hrtf, room=None, refl=0.55, rt60=0.45, reverb_gain=0.18,
                 block=512, volume=0.8, min_dist=0.35, metres_per_unit=1.0):
        self.src, self.fs, self.duration, self.H = sources, fs, duration, hrtf
        self.mpu = metres_per_unit                 # world units -> metres (delay, 1/r)
        self.block, self.volume, self.min_dist = block, volume, min_dist
        self.room = room                           # (min_xyz, max_xyz) or None
        self.refl = refl
        self.pos = 0
        self.playing = True
        self.L = np.zeros(3)
        self.right = np.array([0.0, -1.0, 0.0])
        self.fwd = np.array([1.0, 0.0, 0.0])
        self.lock = threading.Lock()
        self.stream = None
        self.status = "no audio device"
        n_paths = 1 + (6 if room is not None else 0)
        self._prev_idx = [[None] * n_paths for _ in sources]
        self._hist = [[np.zeros(hrtf.taps - 1) for _ in range(n_paths)] for _ in sources]
        self._prev_pose = None
        self.override = None                       # (n, 2) recording played instead of the render
        # Diffuse tail: position-independent, rendered offline once.
        self.tail = None
        if reverb_gain > 0:
            from scipy.signal import oaconvolve
            ir = late_reverb_ir(fs, rt60)
            mix = sum(s.dry[: int(duration * fs)] if len(s.dry) >= int(duration * fs)
                      else np.pad(s.dry, (0, int(duration * fs) - len(s.dry))) for s in sources)
            self.tail = np.stack([oaconvolve(mix, ir[:, c])[: len(mix)] for c in range(2)], 1) * reverb_gain

    # -- transport
    def time(self):
        return self.pos / self.fs

    def seek(self, t):
        with self.lock:
            self.pos = int(max(0.0, t % self.duration) * self.fs)

    def set_listener(self, L, right, fwd):
        self.L, self.right, self.fwd = np.array(L, float), np.array(right, float), np.array(fwd, float)

    def set_traj(self, t, p):
        """Single-source scenes: swap the trajectory driving source 0."""
        self.src[0].set_traj(t, p)

    # -- DSP
    def _read(self, dry, idx):
        N = len(dry)
        idx = np.mod(idx, N - 1)
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        return dry[i0] * (1 - fr) + dry[np.minimum(i0 + 1, N - 1)] * fr

    def block_render(self, n):
        with self.lock:
            p0 = self.pos
            if self.playing:
                self.pos = (self.pos + n) % int(self.duration * self.fs)
        if not self.playing:
            return np.zeros((n, 2), np.float32)
        if self.override is not None:
            i = (p0 + np.arange(n)) % len(self.override)
            return self.override[i].astype(np.float32)
        ts = (p0 + np.arange(n)) / self.fs
        cur = (self.L.copy(), self.right.copy(), self.fwd.copy())
        prev = self._prev_pose or cur
        self._prev_pose = cur
        w = (np.arange(n) / n)[:, None]
        L = prev[0] + (cur[0] - prev[0]) * w
        f = cur[2] / np.linalg.norm(cur[2])
        r = cur[1] / np.linalg.norm(cur[1])
        u = np.cross(r, f)
        to_head = np.stack([f, r, u])                       # world -> head (x fwd, y right, z up)
        fade = np.linspace(0.0, 1.0, n)[:, None]
        out = np.zeros((n, 2))
        taps = self.H.taps
        for si, s in enumerate(self.src):
            S = s.pos(ts)
            paths = [(S, 1.0)]
            if self.room is not None:
                paths += [(I, self.refl) for I in image_sources(S, *self.room)]
            for pi, (P, g) in enumerate(paths):
                v = P - L
                d = np.maximum(np.linalg.norm(v, axis=1) * self.mpu, self.min_dist)
                x = self._read(s.dry, (ts - d / C_SOUND) * self.fs) * (g / d)
                xe = np.concatenate([self._hist[si][pi], x])
                self._hist[si][pi] = xe[-(taps - 1):]
                k = self.H.nearest(to_head @ v[n // 2])
                kp = self._prev_idx[si][pi]
                self._prev_idx[si][pi] = k
                for e in range(2):
                    y = np.convolve(xe, self.H.ir[k, :, e], "valid")
                    if kp is not None and kp != k:
                        yp = np.convolve(xe, self.H.ir[kp, :, e], "valid")
                        y = yp * (1 - fade[:, 0]) + y * fade[:, 0]
                    out[:, e] += y
        if self.tail is not None:
            i = (p0 + np.arange(n)) % len(self.tail)
            out += self.tail[i]
        return np.tanh(self.volume * out).astype(np.float32)

    def start(self):
        try:
            import sounddevice as sd
        except Exception as ex:
            self.status = f"no audio ({ex.__class__.__name__})"
            return False

        def cb(outdata, frames, t, status):
            outdata[:] = self.block_render(frames)

        try:
            self.stream = sd.OutputStream(samplerate=self.fs, channels=2, dtype="float32", blocksize=self.block,
                                          latency="low", callback=cb)
            self.stream.start()
            self.status = f"audio: {sd.query_devices(sd.default.device[1])['name'][:28]}"
            return True
        except Exception as ex:
            self.status = f"no audio ({ex})"[:40]
            return False

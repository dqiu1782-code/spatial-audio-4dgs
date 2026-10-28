"""
simulate.py -- build a simulated scene: several cameras, each with microphones, recording a moving sound source (exact ground truth)

Room: shoebox, optional first-order wall reflections (image-source method, 6 images) + background noise.
Source: moves along a curve and emits speech-like sound (pink noise with syllable modulation + harmonics), or a wav you provide.
Each camera: position + orientation (looking near the room centre) + a mic array (stereo / tetra / hex).
Propagation: per-sample delay from emission to arrival at each mic (moving source -> time-varying delay, so Doppler comes for free), amplitude 1/r.

Output directory: scene.json (cameras, mics, sample rate, ground-truth track) + cam_<name>.wav (multichannel) + source_dry.wav
Usage: python simulate.py --out sim1 --array tetra --cams 4
"""
import argparse
import os

import numpy as np

from geom import ARRAYS, C_SOUND, Camera, look_at, read_wav, save_json, write_wav


def speech_like(n, fs, rng):
    """Coloured noise with syllable modulation (3-6 Hz) + voiced harmonics; spectrum roughly like a voice"""
    t = np.arange(n) / fs
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = 1 / np.sqrt(np.maximum(f, 80)) * (f < 7000)        # pink noise, cut above 7 kHz
    noise = np.fft.irfft(X * shape, n)
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)                  # slowly varying pitch
    phase = 2 * np.pi * np.cumsum(f0) / fs
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    syll = np.clip(np.sin(2 * np.pi * 4.3 * t + 2 * np.sin(2 * np.pi * 0.9 * t)), 0, None) ** 1.5
    pauses = (np.sin(2 * np.pi * 0.25 * t) > -0.7).astype(float)  # occasional pauses
    x = (0.6 * noise / np.std(noise) + 0.5 * voiced / np.std(voiced)) * syll * pauses
    return 0.3 * x / np.max(np.abs(x))


def trajectory(kind, T, room):
    """Returns a function pos(t) (t in seconds, may be an array)"""
    L, W, H = room
    cx, cy = L / 2, W / 2
    if kind == "circle":
        return lambda t: np.stack([cx + 1.6 * np.cos(2 * np.pi * t / T), cy + 1.2 * np.sin(2 * np.pi * t / T),
                                   1.3 + 0.2 * np.sin(4 * np.pi * t / T)], -1)
    if kind == "walk":     # walk a figure eight, pausing in the middle
        def f(t):
            s = 2 * np.pi * (t / T - 0.1 * np.sin(2 * np.pi * t / T) / (2 * np.pi))
            return np.stack([cx + 1.8 * np.sin(s), cy + 1.0 * np.sin(2 * s), 1.6 + 0 * s], -1)
        return f
    if kind == "static":
        return lambda t: np.stack([cx + 0.8 + 0 * t, cy - 0.5 + 0 * t, 1.2 + 0 * t], -1)
    raise ValueError(kind)


def image_sources(p, room):
    """First-order images: mirror once in each of the 6 walls. p: (..., 3) -> (6, ..., 3)"""
    L = np.asarray(room, float)
    out = []
    for ax in range(3):
        for wall in (0.0, L[ax]):
            q = p.copy()
            q[..., ax] = 2 * wall - p[..., ax]
            out.append(q)
    return out


def render_mic(sig, fs, src_pos_fn, mic, n_out, room, beta, order1):
    """Signal received by one mic (per-sample time-varying delay, linear interpolation)"""
    t = np.arange(n_out) / fs
    y = np.zeros(n_out)
    # one iteration for the emission time: t_e = t - |s(t_e) - m| / c
    p = src_pos_fn(t)
    d = np.linalg.norm(p - mic, axis=-1)
    te = t - d / C_SOUND
    p = src_pos_fn(te)
    srcs = [(p, 1.0)]
    if order1:
        srcs += [(q, beta) for q in image_sources(p, room)]
    for q, g in srcs:
        d = np.maximum(np.linalg.norm(q - mic, axis=-1), 0.05)
        idx = (t - d / C_SOUND) * fs
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(sig))
        v = np.zeros(n_out)
        v[ok] = sig[i0[ok]] * (1 - fr[ok]) + sig[i0[ok] + 1] * fr[ok]
        y += g * v / d
    return y


def default_cameras(n, room, array, rng, spread="wide"):
    L, W, H = room
    center = np.array([L / 2, W / 2, 1.3])
    cams = []
    for i in range(n):
        if spread == "wide":       # spread around the walls
            a = 2 * np.pi * i / n + 0.3
            pos = center + np.array([0.45 * L * np.cos(a), 0.45 * W * np.sin(a), 0.0])
        else:                      # clustered together (control)
            pos = np.array([0.4 + 0.3 * i, 0.4, 1.4])
        pos[2] = 1.2 + 0.4 * rng.random()
        tgt = center + rng.normal(0, 0.3, 3)
        cams.append(Camera(f"cam{i + 1}", pos, look_at(pos, tgt), ARRAYS[array]))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim1")
    ap.add_argument("--array", default="tetra", choices=list(ARRAYS))
    ap.add_argument("--cams", type=int, default=4)
    ap.add_argument("--spread", default="wide", choices=["wide", "tight"])
    ap.add_argument("--traj", default="walk", choices=["walk", "circle", "static"])
    ap.add_argument("--dur", type=float, default=12.0, help="seconds")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--room", type=float, nargs=3, default=[7.0, 5.0, 3.0])
    ap.add_argument("--reflect", type=float, default=0.5, help="first-order reflection coefficient (0 = none)")
    ap.add_argument("--snr", type=float, default=25.0, help="dB")
    ap.add_argument("--source_wav", default="", help="use your own mono wav as the source (optional)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fs, n = a.fs, int(a.dur * a.fs)
    if a.source_wav:
        x, sfs = read_wav(a.source_wav)
        x = x.mean(1)
        if sfs != fs:                      # simple linear resampling
            x = np.interp(np.arange(int(len(x) * fs / sfs)) / fs, np.arange(len(x)) / sfs, x)
        sig = np.resize(x, n + fs)
        sig = 0.3 * sig / (np.max(np.abs(sig)) + 1e-9)
    else:
        sig = speech_like(n + fs, fs, rng)
    sig_full = sig
    room = tuple(a.room)
    pos_fn0 = trajectory(a.traj, a.dur, room)
    pos_fn = lambda t: pos_fn0(np.clip(t, 0, a.dur))     # noqa: E731
    cams = default_cameras(a.cams, room, a.array, rng, a.spread)
    for cam in cams:
        chans = []
        for m in cam.mics_world:
            chans.append(render_mic(sig_full, fs, pos_fn, m, n, room, a.reflect, a.reflect > 0))
        Y = np.stack(chans, 1)
        p_sig = np.mean(Y ** 2)
        Y += rng.standard_normal(Y.shape) * np.sqrt(p_sig / 10 ** (a.snr / 10))
        write_wav(os.path.join(a.out, f"{cam.name}.wav"), Y / max(1.0, np.max(np.abs(Y)) * 1.05), fs)
    write_wav(os.path.join(a.out, "source_dry.wav"), sig_full[:n], fs)
    tt = np.arange(0, a.dur, 0.01)
    save_json(os.path.join(a.out, "scene.json"), {
        "fs": fs, "duration": a.dur, "room": list(room), "c": C_SOUND,
        "cameras": [c.to_dict() for c in cams],
        "wav": {c.name: f"{c.name}.wav" for c in cams},
        "time_offset": {c.name: 0.0 for c in cams},
        "truth": {"t": tt.tolist(), "pos": pos_fn(tt).tolist()},
        "sim": vars(a)})
    print(f"simulation done: {a.cams} cameras x {len(ARRAYS[a.array])} mics, {a.dur:.0f} s, {fs} Hz -> {a.out}/")


if __name__ == "__main__":
    main()

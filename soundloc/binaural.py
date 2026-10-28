"""
binaural.py -- render directional binaural audio (for headphones) for a virtual listener from the localised source track

Model: spherical head (Brown & Duda 1998)
  - interaural time difference from the angle of incidence at each ear (Woodworth formula), varying per sample
  - head shadow: first-order shelving filter, the ear facing away loses high frequencies
  - distance: 1/r attenuation + propagation delay (Doppler appears naturally when source/listener move)
  - a little extra high-frequency loss for sources behind, to help front/back (a spherical head alone cannot tell)
No HRTF database needed; a measured HRTF (SOFA file) can be swapped in later.

  python binaural.py sim1                        listener in the middle of the room, head slowly turning a full circle
  python binaural.py sim1 --listener cam2        listener at cam2's position, facing cam2's direction
  python binaural.py sim1 --use truth            render from the ground-truth track (reference)
Output <scene>/binaural_<est|truth>_<listener>.wav
"""
import argparse
import os

import numpy as np

from geom import C_SOUND, load_json, read_wav, write_wav

HEAD_R = 0.0875


def ear_delay(theta):
    """Angle of incidence theta (between source direction and ear axis, radians) -> arrival delay relative to head centre (s)"""
    return np.where(theta < np.pi / 2, -HEAD_R / C_SOUND * np.cos(theta), HEAD_R / C_SOUND * (theta - np.pi / 2))


def shadow_filter(theta, f):
    """Frequency response of the Brown-Duda head-shadow filter (theta in radians, f in Hz)"""
    w0 = C_SOUND / HEAD_R
    alpha = 1.05 + 0.95 * np.cos(np.degrees(theta) / 150.0 * np.pi)
    w = 2 * np.pi * f
    return (1 + 1j * alpha * w / (2 * w0)) / (1 + 1j * w / (2 * w0))


def render(dry, fs, src_fn, lis_fn, yaw_fn, nfft=1024):
    """dry: mono dry signal; src_fn(t)/lis_fn(t) -> (n,3) positions; yaw_fn(t) -> listener heading (radians about z, 0 = +x)"""
    n = len(dry)
    t = np.arange(n) / fs
    S, L, yaw = src_fn(t), lis_fn(t), yaw_fn(t)
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], 1)
    left = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(n)], 1)      # left ear points to +y (when facing +x)
    v = S - L
    r = np.maximum(np.linalg.norm(v, axis=1), 0.3)
    u = v / r[:, None]
    out = np.zeros((n, 2))
    thetas = []
    for e, ear in enumerate((left, -left)):
        th = np.arccos(np.clip(np.sum(u * ear, 1), -1, 1))
        thetas.append(th)
        delay = r / C_SOUND + ear_delay(th)
        idx = (t - delay) * fs
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < n)
        y = np.zeros(n)
        y[ok] = dry[i0[ok]] * (1 - fr[ok]) + dry[i0[ok] + 1] * fr[ok]
        out[:, e] = y / r
    # per-frame head shadow + rear high-frequency loss (STFT overlap-add)
    hop = nfft // 4
    win = np.sqrt(np.hanning(nfft + 1)[:-1])
    f = np.fft.rfftfreq(nfft, 1 / fs)
    behind = np.sum(u * fwd, 1) < 0
    res = np.zeros((n + nfft, 2))
    norm = np.zeros(n + nfft)
    for s in range(0, n - nfft, hop):
        c = s + nfft // 2
        back = np.where(behind[c], 1 / (1 + (f / 5000.0) ** 2) ** 0.25, 1.0)
        for e in range(2):
            Xf = np.fft.rfft(out[s:s + nfft, e] * win)
            Xf *= shadow_filter(thetas[e][c], f) * back
            res[s:s + nfft, e] += np.fft.irfft(Xf, nfft) * win
        norm[s:s + nfft] += win ** 2
    res = res[:n] / np.maximum(norm[:n, None], 1e-6)
    return res


def interp_traj(t_key, P):
    t_key, P = np.asarray(t_key), np.asarray(P)
    return lambda t: np.stack([np.interp(t, t_key, P[:, k]) for k in range(3)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--use", default="est", choices=["est", "truth"])
    ap.add_argument("--listener", default="center", help="center (room centre, head turns a full circle) or a camera name")
    ap.add_argument("--dry", default="", help="dry wav; default: source_dry.wav in simulation, channel 1 of the camera nearest the source for real data")
    a = ap.parse_args()
    S = load_json(os.path.join(a.scene, "scene.json"))
    R = load_json(os.path.join(a.scene, "result.json"))
    fs = S["fs"]
    if a.use == "truth":
        src = interp_traj(S["truth"]["t"], S["truth"]["pos"])
    else:
        src = interp_traj(R["t"], R["smooth"])
    dry_p = a.dry or os.path.join(a.scene, "source_dry.wav")
    if os.path.isfile(dry_p):
        dry, f2 = read_wav(dry_p)
        dry = dry.mean(1)
    else:
        sm = np.array(R["smooth"])
        best = min(S["cameras"], key=lambda c: np.median(np.linalg.norm(sm - np.array(c["pos"]), axis=1)))
        dry, f2 = read_wav(os.path.join(a.scene, S["wav"][best["name"]]))
        dry = dry[:, 0]
        print(f"no dry signal, using channel 1 of {best['name']} instead (includes some room reverb)")
    assert f2 == fs
    if a.listener == "center":
        room = np.array(S.get("room", [6, 5, 3]))
        c = np.array([room[0] / 2, room[1] / 2, 1.6])
        lis = lambda t: np.tile(c, (len(t), 1))                     # noqa: E731
        dur = len(dry) / fs
        yaw = lambda t: 2 * np.pi * t / dur                          # noqa: E731
    else:
        cam = next(c for c in S["cameras"] if c["name"] == a.listener)
        p, fz = np.array(cam["pos"]), np.array(cam["R"])[:, 2]
        lis = lambda t: np.tile(p, (len(t), 1))                     # noqa: E731
        y0 = np.arctan2(fz[1], fz[0])
        yaw = lambda t: np.full(len(t), y0)                          # noqa: E731
    y = render(dry, fs, src, lis, yaw)
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    p = os.path.join(a.scene, f"binaural_{a.use}_{a.listener}.wav")
    write_wav(p, y, fs)
    print(f"binaural audio saved: {p} (listen with headphones)")


if __name__ == "__main__":
    main()

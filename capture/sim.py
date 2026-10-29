"""Capture simulator with ground truth, used to test sync, self-calibration and layering.

Scene: shoebox room, talkers that walk and take turns, background (AC unit, door knocks,
diffuse noise), a 3-clap slate at the stage centre, optional lavaliers with bleed.
Each recorder gets an unknown start time, latency, clock drift, mic offset and rotated mic axis.
Propagation: per-sample time-varying delay (moving sources), 1/r, 6 first-order images, late reverb.
"""
import copy

import numpy as np
from scipy.signal import fftconvolve, lfilter

C = 343.0


def speech_like(n, fs, rng):
    """Speech-like signal: syllabic modulation of pink noise + harmonics with a drifting f0."""
    t = np.arange(n) / fs
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    noise = np.fft.irfft(X / np.sqrt(np.maximum(f, 80)) * (f < 0.45 * fs), n)
    f0 = rng.uniform(110, 210) + 25 * np.sin(2 * np.pi * rng.uniform(0.4, 0.9) * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    voiced = sum(np.sin(k * ph) / k for k in range(1, 12))
    syll = np.clip(np.sin(2 * np.pi * rng.uniform(3.5, 5) * t + 2 * np.sin(2 * np.pi * 0.8 * t)), 0, None) ** 1.5
    x = (0.6 * noise / noise.std() + 0.5 * voiced / voiced.std()) * syll
    return x / np.abs(x).max()


def turn_mask(n, fs, rng, n_talkers, overlap=0.12):
    """Turn-taking gates (n_talkers, n): segments of 1-3.5 s, occasional overlap, short pauses."""
    g = np.zeros((n_talkers, n))
    i, who = int(5.5 * fs), 0                  # claps at 3-4 s, room tone until 5.5 s
    while i < n:
        L = int(rng.uniform(1.0, 3.5) * fs)
        g[who, i:i + L] = 1
        if rng.random() < overlap:
            o = rng.integers(n_talkers)
            if o != who:
                g[o, i + L // 2: i + L] = 1
        i += L + int(rng.uniform(0.1, 0.6) * fs)
        who = (who + rng.integers(1, n_talkers)) % n_talkers if n_talkers > 1 else 0
    k = int(0.02 * fs)                         # 20 ms ramps
    win = np.hanning(2 * k)[:k]
    for a in range(n_talkers):
        d = np.diff(np.concatenate([[0], g[a], [0]]))
        for s in np.nonzero(d > 0)[0]:
            g[a, s:s + k] *= win[: len(g[a, s:s + k])]
        for e in np.nonzero(d < 0)[0]:
            g[a, max(0, e - k):e] *= win[::-1][-len(g[a, max(0, e - k):e]):]
    return g


def smooth_walk(T, rng, centre, spread, z, fs_traj=50):
    """Slow smooth random walk on the stage, returns (t, pos)."""
    t = np.arange(0, T + 1, 1 / fs_traj)
    P = np.zeros((len(t), 3))
    for j in range(2):
        w = rng.standard_normal(len(t))
        k = int(3 * fs_traj)
        w = np.convolve(w, np.hanning(2 * k) / np.hanning(2 * k).sum(), "same")
        w = w / (np.abs(w).max() + 1e-9)
        P[:, j] = centre[j] + spread * w
    P[:, 2] = z + 0.03 * np.sin(2 * np.pi * t / 7.0)
    return t, P


def image_sources(P, room):
    """6 first-order images of P (n, 3) in a box centred on the origin, floor at z=0."""
    half = np.array([room[0] / 2, room[1] / 2])
    walls = [(0, -half[0]), (0, half[0]), (1, -half[1]), (1, half[1]), (2, 0.0), (2, room[2])]
    out = []
    for ax, w in walls:
        Q = P.copy()
        Q[..., ax] = 2 * w - Q[..., ax]
        out.append(Q)
    return out


def render(sig, fs, pos_fn, mic, t_rec, rec_map, room, beta=0.5):
    """Signal at one mic. t_rec: recorder sample times; rec_map(t_rec) -> physical time."""
    tp = rec_map(t_rec)
    P = pos_fn(tp)
    te = tp - np.linalg.norm(P - mic, axis=1) / C          # one fixed-point step for emission time
    P = pos_fn(te)
    y = np.zeros(len(t_rec))
    for Q, g in [(P, 1.0)] + [(q, beta) for q in image_sources(P, room)]:
        d = np.maximum(np.linalg.norm(Q - mic, axis=1), 0.1)
        idx = (tp - d / C) * fs
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(sig))
        v = np.zeros(len(t_rec))
        v[ok] = sig[i0[ok]] * (1 - fr[ok]) + sig[i0[ok] + 1] * fr[ok]
        y += g * v / d
    return y


def tail_ir(fs, rt60, rng, pre=0.03):
    n = int(rt60 * fs)
    t = np.arange(n) / fs
    h = rng.standard_normal(n) * np.exp(-6.91 * t / rt60) * (t > pre)
    return lfilter([0.4], [1, -0.6], h) * 0.03


def perturb_rig(rig, rng, offset_sd=0.04, yaw_sd=25.0, flip_prob=0.25, drift_ppm=20.0):
    """True (unknown) mic parameters: offsets, axis yaw/pitch (some flipped L/R), latency, drift."""
    true = {}
    for m in rig.mics:
        flip = rng.random() < flip_prob and len(m.channels) > 1
        true[m.id] = dict(offset=m.offset + rng.normal(0, offset_sd, 3),
                          yaw=m.axis_yaw + rng.normal(0, yaw_sd) + (180.0 if flip else 0.0),
                          pitch=m.axis_pitch + rng.normal(0, 8.0),
                          latency=float(np.clip(rng.normal(0, m.latency_sd), -3 * m.latency_sd, 3 * m.latency_sd)),
                          drift=float(rng.normal(0, drift_ppm) * 1e-6),
                          start=float(rng.uniform(0.0, 2.0)))          # recorder started this much late
    return true


def simulate(rig, dur=40.0, fs=16000, seed=0, n_talkers=2, snr_db=30.0, rt60=0.45, lav=True):
    rng = np.random.default_rng(seed)
    room = rig.room.get("size_m", [7.0, 6.0, 2.8])
    n = int(dur * fs)
    tphys = np.arange(n) / fs
    # ---- talkers
    talkers = []
    gates = turn_mask(n, fs, rng, n_talkers)
    for a in range(n_talkers):
        c0 = rng.uniform(-0.5, 0.5, 2)
        tt, P = smooth_walk(dur, rng, c0, 0.45, rng.uniform(1.45, 1.70))
        x = speech_like(n, fs, rng) * gates[a]
        talkers.append(dict(name=f"T{a + 1}", sig=x, t=tt, pos=P,
                            fn=lambda q, tt=tt, P=P: np.stack([np.interp(q, tt, P[:, j]) for j in range(3)], 1)))
    # ---- background: AC unit (fixed, low pink noise), door knocks, claps at the stage centre
    ac_pos = np.array([room[0] / 2 - 0.3, room[1] / 4, 2.2])
    ac = lfilter([1], [1, -0.97], rng.standard_normal(n))
    ac = 0.04 * ac / ac.std()
    door_pos = np.array([-room[0] / 2 + 0.2, -room[1] / 2 + 1.0, 1.0])
    door = np.zeros(n)
    for tk in rng.uniform(8, dur - 2, 3):
        for j in range(3):
            i = int((tk + 0.18 * j) * fs)
            door[i:i + 200] += rng.standard_normal(min(200, n - i)) * np.exp(-np.arange(min(200, n - i)) / 40)
    clap = np.zeros(n)
    clap_times = [3.0, 3.5, 4.0]                  # after every recorder has started (start < 2 s)
    for tc in clap_times:
        i = int(tc * fs)
        clap[i:i + 160] += rng.standard_normal(160) * np.exp(-np.arange(160) / 25) * 3.0
    static = lambda p: (lambda q: np.repeat(p[None], len(q), 0))    # noqa: E731
    bg_sources = [("ac", ac, static(ac_pos)), ("door", door * 0.5, static(door_pos)),
                  ("clap", clap, static(np.array([0.0, 0.0, 1.5])))]
    # ---- recorders
    true = perturb_rig(rig, rng)
    mics, voice, backg = {}, {}, {}
    irs = [tail_ir(fs, rt60, rng) for _ in range(4)]
    for m in rig.mics:
        tr = true[m.id]
        P = rig.channel_positions_world(m, offset=tr["offset"], yaw=tr["yaw"], pitch=tr["pitch"])
        t_rec = np.arange(n) / fs
        rec_map = lambda t, tr=tr: tr["start"] + t * (1 + tr["drift"]) - tr["latency"]   # noqa: E731
        Y, V, B = (np.zeros((n, len(P))) for _ in range(3))
        for k, p in enumerate(P):
            for s in talkers:
                V[:, k] += render(s["sig"], fs, s["fn"], p, t_rec, rec_map, room)
            for name, sig, fn in bg_sources:
                B[:, k] += render(sig, fs, fn, p, t_rec, rec_map, room)
            # late reverb (decorrelated per channel) of every source + diffuse room noise
            tail = sum(fftconvolve(s["sig"], irs[(k + i) % 4])[:n] for i, s in enumerate(talkers))
            V[:, k] += np.interp(rec_map(t_rec), tphys, tail)
            tail_b = fftconvolve(ac + door * 0.5, irs[(k + 3) % 4])[:n]
            B[:, k] += np.interp(rec_map(t_rec), tphys, tail_b)
        diffuse = lfilter([1], [1, -0.9], rng.standard_normal((n, len(P))), axis=0)
        B += 0.004 * diffuse / diffuse.std()
        sig_rms = np.sqrt(np.mean(V ** 2))
        Y = V + B + rng.standard_normal(V.shape) * sig_rms * 10 ** (-snr_db / 20)
        mics[m.id], voice[m.id], backg[m.id] = Y.astype(np.float32), V.astype(np.float32), B.astype(np.float32)
    lavs = {}
    if lav:
        for a, s in enumerate(talkers):
            lat = rng.uniform(0.0, 0.003)
            other = sum(o["sig"] for o in talkers if o is not s)
            x = np.interp(tphys - lat, tphys, s["sig"]) + 0.08 * np.interp(tphys - lat - 0.004, tphys, other)
            d_clap = np.linalg.norm(s["fn"](np.array([clap_times[0]]))[0] - np.array([0.0, 0.0, 1.5]))
            x = x + 0.3 / max(d_clap, 0.3) * np.interp(tphys - lat - d_clap / C, tphys, clap)
            x = x + 0.003 * rng.standard_normal(n)
            lavs[s["name"]] = x.astype(np.float32)
            true[f"lav:{s['name']}"] = dict(latency=lat)
    truth = dict(talkers=[dict(name=s["name"], t=s["t"], pos=s["pos"]) for s in talkers], mics=true,
                 clap_times=clap_times, clap_pos=[0.0, 0.0, 1.5], room=room, fs=fs, dur=dur)
    return dict(fs=fs, mics=mics, lavs=lavs, truth=truth, voice=voice, background=backg)


def cached_simulation(rig, rig_path, **kw):
    """Simulate once per rig file; cache under $SA4DGS_CACHE or <package>/cache."""
    import os
    import pickle
    root = os.environ.get("SA4DGS_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    os.makedirs(root, exist_ok=True)
    import hashlib
    import json
    acoustic = {k: rig.raw.get(k) for k in ("cameras", "lavaliers", "room", "stage", "sync")}   # what the audio depends on
    key = hashlib.sha1(json.dumps([acoustic, kw], sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = os.path.join(root, f"sim_{os.path.basename(rig_path)}_{key}.pkl")   # new rig -> new simulation
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    S = simulate(rig, **{"dur": 40.0, "fs": 16000, "seed": 1, **kw})
    with open(path, "wb") as f:
        pickle.dump(S, f)
    return S

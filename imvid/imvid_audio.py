"""Sound pipeline on the ImViD Opera sample (39 GoPro stereo recordings), metric world from imvid_finalize.py.

1. Sync: the singer's voice onset is used as the slate (no claps were recorded). Arrival time per
   recorder (onset envelope + GCC-PHAT, capture/sync.py) minus propagation from the mouth (video track)
   gives each recorder's offset.
2. Audio-only localisation: SRP-PHAT over all mic pairs on a 3D grid (coarse 10 cm, fine 2 cm), per
   0.25 s window, offsets compensated. Compared with the video mouth track.
3. Dry voice: delay-and-sum beamformer over all 39 recorders steered at the source.
4. Sound scene for braindance_viewer.py (scene.json, result.json, FLACs).

  python imvid_audio.py DATA AUDIO_DIR [--win_src E:\\4dgs_data\\imvid_opera\\sound]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

from scipy.signal import butter, sosfiltfilt

C = 343.0


def gcc_phat_all(Y, fs, maxlag):
    """Y: (n_mic, n) -> (pairs (i, j), cc (n_pairs, 2*maxlag+1)) for lag = t_j - t_i."""
    n = 1 << int(np.ceil(np.log2(2 * Y.shape[1])))
    F = np.fft.rfft(Y * np.hanning(Y.shape[1]), n)
    F /= np.abs(F) + 1e-9
    I, J = np.triu_indices(len(Y), 1)
    cc = np.fft.irfft(F[J] * np.conj(F[I]), n)
    cc = np.concatenate([cc[:, -maxlag:], cc[:, : maxlag + 1]], 1)
    return I, J, cc


def srp(I, J, cc, fs, maxlag, mics, grid):
    d = np.linalg.norm(grid[:, None, :] - mics[None], axis=2) / C       # (g, m)
    lag = (d[:, J] - d[:, I]) * fs + maxlag                              # (g, p)
    lo = np.clip(np.floor(lag).astype(int), 0, 2 * maxlag - 1)
    w = np.clip(lag - lo, 0, 1)
    p = np.arange(len(I))[None]
    v = cc[p, lo] * (1 - w) + cc[p, lo + 1] * w
    return v.sum(1)


def frac_shift(x, s, fs):
    """x(t + s) with fractional s (seconds), FFT based."""
    n = 1 << int(np.ceil(np.log2(len(x) + int(abs(s) * fs) + 16)))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(X * np.exp(2j * np.pi * f * s), n)[: len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("audio")
    ap.add_argument("--win_src", default=r"E:\4dgs_data\imvid_opera\sound")
    ap.add_argument("--onset_search", nargs=2, type=float, default=(2.0, 3.5))
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "braindance_meta.json")))
    assert meta.get("metric"), "run imvid_finalize.py first"
    tr = json.load(open(os.path.join(a.data, "face_track.json")))
    names = sorted(meta["cameras"])
    mics = np.array([meta["cameras"][n]["C"] for n in names])
    tv, mouth = np.array(tr["t"]), np.array(tr["mouth"])
    raw = {}
    for n in names:
        fs, x = wavfile.read(os.path.join(a.audio, n + ".wav"))
        raw[n] = x.astype(np.float64) / 32768.0
    mono = np.array([raw[n].mean(1) for n in names])
    dur = mono.shape[1] / fs

    # ---- 1. onset slate sync: the voice attack (+15 dB within a few ms) on every recorder
    sos = butter(4, [200, 6000], "bandpass", fs=fs, output="sos")
    bp = sosfiltfilt(sos, mono, axis=1)
    hop, w = int(0.00025 * fs), int(0.001 * fs)
    cross = []
    for i in range(len(names)):
        p2 = np.convolve(bp[i] ** 2, np.ones(w) / w, "same")[::hop]
        db = 10 * np.log10(p2 + 1e-14)
        tt_ = np.arange(len(db)) * hop / fs
        nf = np.median(db[(tt_ > 0.3) & (tt_ < 1.4)])
        k = int(np.argmax((db > nf + 15) & (tt_ > a.onset_search[0]) & (tt_ < a.onset_search[1])))
        k0 = max(k - 1, 0)
        frac = (nf + 15 - db[k0]) / (db[k] - db[k0] + 1e-9) if k > 0 else 0.0
        cross.append(tt_[k0] + np.clip(frac, 0, 1) * hop / fs)
    cross = np.array(cross)
    ref = int(np.argmin(np.linalg.norm(mics - np.median(mouth, 0), axis=1)))
    t_on = float(cross[ref])

    def gcc_near(i):
        """Arrival in recorder i: GCC-PHAT on the 50 ms attack, lag searched within 1 ms of the envelope estimate."""
        if i == ref:
            return t_on
        L = int(0.05 * fs)
        r = bp[ref, int((t_on - 0.005) * fs): int((t_on - 0.005) * fs) + L]
        x = bp[i, int((cross[i] - 0.005) * fs): int((cross[i] - 0.005) * fs) + L]
        n = 1 << int(np.ceil(np.log2(2 * L)))
        X = np.fft.rfft(x * np.hanning(L), n) * np.conj(np.fft.rfft(r * np.hanning(L), n))
        cc = np.fft.irfft(X / (np.abs(X) + 1e-12), n)
        m = int(0.001 * fs)
        cc = np.concatenate([cc[-m:], cc[: m + 1]])
        k = int(np.argmax(cc))
        if 0 < k < len(cc) - 1:
            y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        return cross[i] + (k - m) / fs

    arr = np.array([gcc_near(i) for i in range(len(names))])
    print(f"voice attack: envelope vs GCC refinement differ by median {np.median(np.abs(arr - cross)) * 1e3:.2f} ms")
    X_on = np.array([np.interp(t_on, tv, mouth[:, k]) for k in range(3)])
    prop = np.linalg.norm(mics - X_on, axis=1) / C
    off = (arr - prop) - (arr[ref] - prop[ref])                          # t_rec = t_phys + off
    print(f"slate: voice attack at {t_on:.4f} s on {names[ref]} (closest to the singer)")
    print(f"recorder offsets: median |off| {np.median(np.abs(off)) * 1e3:.2f} ms, range "
          f"{off.min() * 1e3:+.2f} .. {off.max() * 1e3:+.2f} ms; propagation spread {np.ptp(prop) * 1e3:.2f} ms")
    ok = np.abs(off) < 0.15                                               # gross failures (no clear attack)
    if (~ok).any():
        print("excluded recorders:", [names[i] for i in np.nonzero(~ok)[0]])
    al = np.array([frac_shift(mono[i], off[i], fs) for i in range(len(names))])   # aligned to physical time
    al_all, mics_all = al, mics
    al, mics = al[ok], mics[ok]

    # ---- 2. SRP-PHAT localisation
    maxlag = int(0.006 * fs)
    gx = np.arange(-2.0, 2.01, 0.1)
    gy = np.arange(-2.0, 2.01, 0.1)
    gz = np.arange(0.3, 2.21, 0.1)
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    win, hop = 0.25, 0.125
    rows = []
    for t0 in np.arange(t_on, dur - win, hop):
        seg = al[:, int(t0 * fs): int((t0 + win) * fs)]
        if np.sqrt((seg ** 2).mean()) < 3 * np.sqrt((al[:, int(0.3 * fs): int(1.8 * fs)] ** 2).mean()):
            continue
        I, J, cc = gcc_phat_all(seg, fs, maxlag)
        best = G[np.argmax(srp(I, J, cc, fs, maxlag, mics, G))]
        f = np.arange(-0.15, 0.151, 0.02)
        G2 = best + np.stack(np.meshgrid(f, f, f, indexing="ij"), -1).reshape(-1, 3)
        x = G2[np.argmax(srp(I, J, cc, fs, maxlag, mics, G2))]
        tc = t0 + win / 2
        xv = np.array([np.interp(tc, tv, mouth[:, k]) for k in range(3)])
        rows.append((tc, x, xv))
    ta = np.array([r[0] for r in rows])
    Xa = np.array([r[1] for r in rows])
    Xv = np.array([r[2] for r in rows])
    e3 = np.linalg.norm(Xa - Xv, axis=1) * 100
    eh = np.linalg.norm((Xa - Xv)[:, :2], axis=1) * 100
    cen = mics.mean(0)
    ang = np.degrees(np.arccos(np.clip(np.sum((Xa - cen) * (Xv - cen), 1) /
                                       np.linalg.norm(Xa - cen, axis=1) / np.linalg.norm(Xv - cen, axis=1), -1, 1)))
    print(f"audio-only localisation: {len(rows)} windows; vs video mouth: 3D median {np.median(e3):.1f} cm "
          f"(p90 {np.percentile(e3, 90):.1f}), horizontal {np.median(eh):.1f} cm, direction from rig centre "
          f"{np.median(ang):.1f} deg, range to rig {np.median(np.linalg.norm(Xv - cen, axis=1)):.2f} m")
    # without sync (raw recordings) for comparison
    rows0 = []
    for t0 in ta - win / 2:
        seg = mono[ok][:, int(t0 * fs): int((t0 + win) * fs)]
        I, J, cc = gcc_phat_all(seg, fs, maxlag)
        rows0.append(G[np.argmax(srp(I, J, cc, fs, maxlag, mics, G))])
    e0 = np.linalg.norm(np.array(rows0) - Xv, axis=1) * 100
    print(f"  same without the slate sync: 3D median {np.median(e0):.1f} cm")

    # ---- 3. delay-and-sum dry voice (steered at the video mouth track's median; the singer barely moves)
    Xs = np.median(mouth, 0)
    d = np.linalg.norm(mics - Xs, axis=1)
    dry = np.mean([frac_shift(al[i], d[i] / C, fs) * d[i] / d.mean() for i in range(len(al))], 0)
    sil = slice(int(0.3 * fs), int(1.8 * fs))
    act = slice(int((t_on + 0.2) * fs), int((dur - 0.1) * fs))
    snr = lambda x: 10 * np.log10((x[act] ** 2).mean() / (x[sil] ** 2).mean())       # noqa: E731
    print(f"dry voice: voice-to-room-noise {snr(mono[ref]):.1f} dB (closest single mic) -> {snr(dry):.1f} dB (39-mic beamformer)")

    # ---- 4. sound scene for the viewer
    sd = os.path.join(a.data, "sound")
    os.makedirs(sd, exist_ok=True)
    wavfile.write(os.path.join(sd, "singer_dry.wav"), fs, (dry / np.abs(dry).max() * 0.5 * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(sd, "singer_dry.wav"),
                    os.path.join(sd, "singer_dry.flac")], check=True)
    os.remove(os.path.join(sd, "singer_dry.wav"))
    for n in names:
        src = os.path.join(a.audio, n + ".flac")
        if os.path.isfile(src):
            subprocess.run(["cp", src, os.path.join(sd, n + ".flac")], check=True)
    tt = np.round(np.arange(0, dur + 1e-9, 0.05), 3)
    pos_v = np.stack([np.interp(tt, tv, mouth[:, k]) for k in range(3)], 1)
    if len(ta) >= 3:
        from scipy.signal import medfilt
        Xm = np.stack([medfilt(Xa[:, k], 5) for k in range(3)], 1)
        pos_a = np.stack([np.interp(tt, ta, Xm[:, k]) for k in range(3)], 1)
    else:
        pos_a = pos_v
    cams = [{"name": n, "pos": mics_all[i].tolist(), "binaural": n + ".flac", "ear_axis": np.array(meta["cameras"][n]["R"])[:, 0].tolist()}
            for i, n in enumerate(names)]
    room = [[-2.5, -4.5, 0.0], [4.5, 4.5, 4.5]]          # stage box (metres): curtain ~2 m behind the singer
    json.dump({"src_dir": a.win_src, "clip_start": 0.0, "duration": round(dur, 3), "cameras": cams, "room": room,
               "dataset": "ImViD scene1_opera sample (CC BY 4.0)"}, open(os.path.join(sd, "scene.json"), "w"), indent=1)
    json.dump({"metres_per_unit": 1.0, "audio_offset_s": 0.0, "clip_start": 0.0,
               "sources": [{"name": "singer", "dry": "singer_dry.flac", "t": tt.tolist(), "pos": pos_a.tolist(),
                            "pos_video": pos_v.tolist()}],
               "t": tt.tolist(), "smooth": pos_a.tolist(),
               "heldout_err_cm": {"median_3d": float(np.median(e3)), "median_horizontal": float(np.median(eh)),
                                  "median_direction_deg": float(np.median(ang)), "without_sync_3d": float(np.median(e0))},
               "sync": {"slate": "voice onset", "t_onset": t_on, "ref": names[ref],
                        "offsets_ms": {n: float(off[i] * 1e3) for i, n in enumerate(names)},
                        "excluded": [names[i] for i in np.nonzero(~ok)[0]]}},
              open(os.path.join(sd, "result.json"), "w"), indent=1)
    print("AUDIO_DONE ->", sd)


if __name__ == "__main__":
    main()

"""
run_localize.py -- main program: per-camera sound direction -> multi-camera triangulation -> smoothed track -> (with ground truth) evaluation + plots

  python run_localize.py sim1                 (sim1/ is output of simulate.py, or a real scene written in the same format)
  python run_localize.py sim1 --hop 0.05 --nfft 4096

The scene directory needs scene.json (camera poses + mic positions per camera + wav file names + per-camera time offsets) and each camera's wav.
Output: <scene>/result.json, <scene>/fig_*.png (only if matplotlib is installed)
"""
import argparse
import os
import time

import numpy as np

from doa import DOAEstimator
from geom import Camera, load_json, read_wav, save_json
from triangulate import kalman_smooth, solve


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--nfft", type=int, default=4096, help="samples per frame (4096 ~= 85 ms at 48 kHz)")
    ap.add_argument("--hop", type=float, default=0.05, help="frame hop (s)")
    ap.add_argument("--fmin", type=float, default=200)
    ap.add_argument("--fmax", type=float, default=8000)
    ap.add_argument("--vad", type=float, default=0.25, help="skip a frame as silent if its energy is below this fraction of the camera's median")
    ap.add_argument("--min_cams", type=int, default=2)
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args()
    S = load_json(os.path.join(a.scene, "scene.json"))
    fs = S["fs"]
    cams = [Camera.from_dict(d) for d in S["cameras"]]
    X = {}
    for c in cams:
        x, f2 = read_wav(os.path.join(a.scene, S["wav"][c.name]))
        assert f2 == fs, f"{c.name} sample rate {f2} != {fs}"
        assert x.shape[1] == len(c.mics_local), f"{c.name}: wav has {x.shape[1]} channels but scene.json lists {len(c.mics_local)} mics"
        off = int(round(S.get("time_offset", {}).get(c.name, 0.0) * fs))   # how many seconds after the reference clock this camera started
        X[c.name] = np.vstack([np.zeros((max(off, 0), x.shape[1])), x[max(-off, 0):]])
    n = min(len(v) for v in X.values())
    hop = int(a.hop * fs)
    starts = np.arange(0, n - a.nfft, hop)
    t = (starts + a.nfft / 2) / fs
    est = {c.name: DOAEstimator(c.mics_local, fs, a.nfft, fmin=a.fmin, fmax=a.fmax) for c in cams}
    kinds = {c.name: ("full 3D direction" if est[c.name].M > 2 and est[c.name].rank >= 2 else "cone (angle only)") for c in cams}
    print("cameras: " + ", ".join(f"{c.name} ({len(c.mics_local)} mics, {kinds[c.name]})" for c in cams))
    energy = {c.name: np.array([np.mean(X[c.name][s:s + a.nfft] ** 2) for s in starts]) for c in cams}
    room = S.get("room")
    if "bounds" in S:
        bounds = S["bounds"]
    elif room:
        bounds = ([0, 0, 0], room)
    else:
        P = np.array([c.pos for c in cams])
        span = np.ptp(P, 0).max()
        bounds = (P.min(0) - span, P.max(0) + span)
    t0 = time.perf_counter()
    raw = np.full((len(t), 3), np.nan)
    resid = np.full(len(t), np.nan)
    ncam = np.zeros(len(t), int)
    doas = {c.name: [] for c in cams}
    for k, s in enumerate(starts):
        obs = []
        for c in cams:
            e = energy[c.name]
            if e[k] < a.vad * np.median(e):
                doas[c.name].append(None)
                continue
            r = est[c.name].estimate(X[c.name][s:s + a.nfft])
            if r["kind"] == "dir":
                d = c.R @ r["u"]
                obs.append({"kind": "dir", "c": c.pos, "d": d, "w": max(r["score"], 0.05)})
                doas[c.name].append({"kind": "dir", "d": d.tolist(), "score": r["score"]})
            else:
                ax = c.R @ r["axis"]
                obs.append({"kind": "cone", "c": c.pos, "axis": ax, "cos": r["cos"], "w": max(r["score"], 0.05)})
                doas[c.name].append({"kind": "cone", "axis": ax.tolist(), "cos": r["cos"], "score": r["score"]})
        need = a.min_cams if all(o["kind"] == "dir" for o in obs) else max(a.min_cams, 3)
        if len(obs) >= need:
            p0 = raw[k - 1] if k and not np.isnan(raw[k - 1, 0]) else None
            raw[k], resid[k] = solve(obs, bounds, p0=p0)
            resid[k] = np.degrees(resid[k])
            ncam[k] = len(obs)
        if k % 40 == 0:
            print(f"  frame {k}/{len(starts)}", flush=True)
    valid = ~np.isnan(raw[:, 0]) & (resid < 15)
    sc = S.get("scale", 1.0)          # metres per scene unit (1 in simulation)
    smooth = kalman_smooth(t, np.nan_to_num(raw), valid, q=2.0 / sc, r=0.15 / sc)
    print(f"processed {len(t)} frames ({n / fs:.1f} s of audio) in {time.perf_counter() - t0:.1f} s; {valid.sum()} valid frames")
    out = {"t": t.tolist(), "raw": np.where(np.isnan(raw), None, raw).tolist(), "smooth": smooth.tolist(),
           "valid": valid.tolist(), "resid_deg": np.where(np.isnan(resid), None, resid).tolist(),
           "n_cams": ncam.tolist(), "doa": doas}
    if "truth" in S:
        tt, pp = np.array(S["truth"]["t"]), np.array(S["truth"]["pos"])
        gt = np.stack([np.interp(t, tt, pp[:, k]) for k in range(3)], 1)
        e_raw = np.linalg.norm(raw - gt, axis=1)[valid]
        e_sm = np.linalg.norm(smooth - gt, axis=1)
        # per-camera direction error (vs ground-truth direction; for cones, vs the angle)
        dir_err = {}
        for c in cams:
            errs = []
            for k, dd in enumerate(doas[c.name]):
                if dd is None:
                    continue
                v = gt[k] - c.pos
                if dd["kind"] == "dir":
                    errs.append(angle_deg(dd["d"], v))
                else:
                    errs.append(abs(angle_deg(dd["axis"], v) - np.degrees(np.arccos(dd["cos"]))))
            dir_err[c.name] = float(np.median(errs)) if errs else None
        out["eval"] = {"raw_median_m": float(np.median(e_raw)), "raw_p90_m": float(np.percentile(e_raw, 90)),
                       "smooth_median_m": float(np.median(e_sm)), "smooth_p90_m": float(np.percentile(e_sm, 90)),
                       "doa_median_deg": dir_err}
        out["gt"] = gt.tolist()
        print("localisation error: per-frame median {:.3f} m (90th pct {:.3f}), smoothed median {:.3f} m (90th pct {:.3f})".format(
            out["eval"]["raw_median_m"], out["eval"]["raw_p90_m"], out["eval"]["smooth_median_m"],
            out["eval"]["smooth_p90_m"]))
        print("per-camera median direction error (deg): " + ", ".join(f"{k} {v:.1f}" for k, v in dir_err.items() if v is not None))
    save_json(os.path.join(a.scene, "result.json"), out)
    if not a.no_plot:
        try:
            import plot_result
            plot_result.plot(a.scene, S, out)
        except ImportError as e:
            print("no plots (matplotlib missing):", e)


if __name__ == "__main__":
    main()

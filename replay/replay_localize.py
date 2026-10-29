"""Audio-only localisation of each actor for a prepared Replay clip.

Uses the sync/calibration from replay_sync.py (audio offset, metres-per-unit, latency, dz).
Per voiced window: lavalier->ear ranges + per-recorder interaural delays -> robust 3D fit over 8 binaural mics.
Trajectories are median-filtered and held between voiced windows.

  python replay_localize.py E:\\4dgs_data\\replay_1044
Writes <data>/sound/result.json and <data>/overlay_localization.png (audio: solid, video: thin).
"""
import json
import math
import os
import sys

import numpy as np
from scipy.signal import medfilt

from replay_common import AudioScene, load_cameras, solve_window

INITS = [np.array(v) for v in ([0, 0, -0.3], [0.4, 0, -0.3], [-0.4, 0, -0.3], [0, 0.4, -0.3], [0, -0.4, -0.3])]


def main():
    data = sys.argv[1]
    sd = os.path.join(data, "sound")
    S = json.load(open(os.path.join(sd, "scene.json")))
    meta = json.load(open(os.path.join(data, "scene_meta.json")))
    src = S["src_dir"]
    sync = json.load(open(os.path.join(src, "sync.json")))
    t0, dur = S["clip_start"], S["duration"]
    off, mpu, dz = sync["audio_offset_s"], sync["metres_per_unit"], sync["source_dz_units"]
    dl = np.atleast_1d(np.array(sync["latency_s"], float))
    bI, cI = np.array(sync["itd_gain_s"]), np.array(sync["itd_bias_s"])
    eps = sync.get("lavalier_latency_s", {})
    cams, _, _ = load_cameras(meta["seq"])
    A = AudioScene(src, cams)
    obs = A.observations(t0 + off - 0.5, t0 + off + dur + 0.5, hop=0.1)
    print(f"{len(obs)} voiced windows in clip", flush=True)

    tt = np.arange(0, dur + 1e-9, 0.05)
    sources, report = [], []
    for a in A.refs:
        rows = []
        X0 = np.array([0.0, 0.0, -0.3])
        for o in obs:
            if o[0] != a:
                continue
            good = o[3] > 4.0
            if good.sum() < 5:
                continue
            X, r = None, np.inf
            for x0 in [X0] + INITS:                           # previous estimate + generic starts
                Xi, ri = solve_window(o[2] - eps.get(a, 0.0), good, A.Cm, mpu, dl, X0=x0, itd=o[4] - cI, gi=o[5] > 4.0,
                                      Rm=A.Rm, b=bI)
                if ri < r:
                    X, r = Xi, ri
            if r < 0.6:
                rows.append((o[1] - off - t0, X))
                X0 = X
        pos_video = None
        if a in sync["visual"]:
            tv = np.array(sync["visual"][a]["t"]) - t0
            Xv = np.array(sync["visual"][a]["pos"]) + np.array([0, 0, dz])
            if ((tv >= 0) & (tv <= dur)).sum() > 10:
                pos_video = np.stack([np.interp(tt, tv, Xv[:, j]) for j in range(3)], 1)
        if len(rows) < 3:
            print(f"  {a}: too few reliable audio windows ({len(rows)})"
                  + ("; using the video track" if pos_video is not None else "; skipped"))
            if pos_video is None:
                continue
            sources.append({"name": a, "dry": a + ".flac", "t": tt.tolist(), "pos": pos_video.tolist(),
                            "pos_video": pos_video.tolist(), "raw_t": [], "raw_pos": []})
            continue
        ta = np.array([r[0] for r in rows])
        Xa = np.array([r[1] for r in rows])
        ks = min(7, len(Xa) // 2 * 2 - 1)
        Xs = np.stack([medfilt(Xa[:, j], ks) for j in range(3)], 1) if ks >= 3 else Xa
        pos = np.stack([np.interp(tt, ta, Xs[:, j]) for j in range(3)], 1)
        line = f"  {a}: {len(rows)} windows"
        if a in sync["visual"]:
            tv = np.array(sync["visual"][a]["t"]) - t0
            Xv = np.array(sync["visual"][a]["pos"]) + np.array([0, 0, dz])
            m = (tv >= 0) & (tv <= dur)
            if m.any():
                P = np.stack([np.interp(tv[m], tt, pos[:, j]) for j in range(3)], 1)
                e = np.linalg.norm(P - Xv[m], axis=1) * mpu * 100
                line += f", vs video: median {np.median(e):.1f} cm over {m.sum()} frames"
                report.append(float(np.median(e)))
        print(line, flush=True)
        src_entry = {"name": a, "dry": a + ".flac", "t": tt.tolist(), "pos": pos.tolist(),
                     "raw_t": ta.tolist(), "raw_pos": Xa.tolist()}
        if pos_video is not None:
            src_entry["pos_video"] = pos_video.tolist()
        sources.append(src_entry)
    res = {"metres_per_unit": mpu, "latency_s": dl.tolist(), "audio_offset_s": off, "clip_start": t0,
           "sources": sources, "t": tt.tolist(), "smooth": sources[0]["pos"] if sources else [],
           "clip_err_vs_video_cm": report, "heldout_err_cm": sync.get("heldout_err_cm")}
    json.dump(res, open(os.path.join(sd, "result.json"), "w"), indent=1)

    # ---- overlay: audio estimate (thick) and video track (thin) in four views
    from PIL import Image, ImageDraw
    tj = json.load(open(os.path.join(data, "transforms_train.json")))
    W, H = tj["w"], tj["h"]
    fx = W / (2 * math.tan(tj["camera_angle_x"] / 2))
    by_cam = {}
    for f in tj["frames"]:
        by_cam.setdefault(os.path.basename(f["file_path"]).split("_t")[0], []).append(f)
    cols = [(255, 60, 40), (40, 150, 255), (60, 200, 80), (240, 200, 40)]
    tiles = []
    for cam in list(by_cam)[:4]:
        for frac in (0.2, 0.5, 0.8):
            fr = by_cam[cam][int(frac * (len(by_cam[cam]) - 1))]
            tsec = fr["time"] * dur
            im = Image.open(os.path.join(data, fr["file_path"].lstrip("./") + ".png")).convert("RGB")
            dr = ImageDraw.Draw(im)
            M = np.array(fr["transform_matrix"])
            Rcv, C = M[:3, :3] * np.array([1, -1, -1]), M[:3, 3]

            def draw(p, col, width):
                q = Rcv.T @ (p - C)
                if q[2] > 0.05:
                    u, v = fx * q[0] / q[2] + W / 2, fx * q[1] / q[2] + H / 2
                    rad = max(8, int(0.12 / mpu * fx / q[2]))
                    dr.ellipse([u - rad, v - rad, u + rad, v + rad], outline=col, width=width)
            for si, s in enumerate(sources):
                p = np.array([np.interp(tsec, s["t"], np.array(s["pos"])[:, j]) for j in range(3)])
                draw(p, cols[si % 4], 5)
                if s["name"] in sync["visual"]:
                    tv = np.array(sync["visual"][s["name"]]["t"]) - t0
                    Xv = np.array(sync["visual"][s["name"]]["pos"]) + np.array([0, 0, dz])
                    if len(tv) and np.min(np.abs(tv - tsec)) < 0.2:
                        draw(np.array([np.interp(tsec, tv, Xv[:, j]) for j in range(3)]), cols[si % 4], 1)
            dr.text((8, 8), f"{cam}  t={tsec:.1f}s", fill=(255, 255, 255))
            tiles.append(im.resize((W // 2, H // 2)))
    grid = Image.new("RGB", (3 * W // 2, 4 * H // 2))
    for k, im in enumerate(tiles):
        grid.paste(im, ((k % 3) * W // 2, (k // 3) * H // 2))
    grid.save(os.path.join(data, "overlay_localization.png"))
    print("LOC_DONE ->", os.path.join(data, "overlay_localization.png"))


if __name__ == "__main__":
    main()

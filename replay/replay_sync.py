"""Audio/video sync and audio-model calibration for a Replay scene.

1. Visual track: triangulate each actor's head point from the 8 DSLR boxes (video-local time);
   metric scale from the actors' standing height (box top/bottom, assumed 1.72 m on average).
2. Audio: per voiced window, lavalier->ear delays and each recorder's interaural delay (ITD).
3. Sync: from the metadata (timestamp_delta = global time of a file's first sample), cross-checked
   against the interaural-delay cue (offset-free: both ears share one recorder).
4. Calibrate on even seconds: ITD gains, per-recorder and per-lavalier latencies, source height
   offset; report audio-only localisation error (ranges + ITD) on odd seconds (held out).

  python replay_sync.py SC-1044
Writes E:\\replay\\<seq>\\sync.json
"""
import json
import os
import sqlite3
import sys
import time

import numpy as np

from replay_common import DB, C_SOUND, AudioScene, cos_to_ear_axis, load_cameras, solve_window, triangulate

H_ASSUMED = 1.72     # m, mean standing height of the actors (box top to bottom)


def visual_tracks(db, seq, cams, delta_v, head_frac=0.09):
    rows = db.execute(
        "SELECT a.actor_id, f.sensor_name, f.frame_timestamp, a.bounding_box_xywh FROM actor_frame_annots a "
        "JOIN frame_annots f ON a.sequence_name=f.sequence_name AND a.frame_number=f.frame_number "
        "WHERE a.sequence_name=? AND a.bounding_box_xywh IS NOT NULL AND f.sensor_name LIKE 'DSLR-%'",
        (seq,)).fetchall()
    groups = {}
    for actor, sensor, ts, bb in rows:
        x, y, w, h = np.frombuffer(bb, "<f4", count=4)
        groups.setdefault((actor, round(ts * 30)), {})[sensor] = (x, y, w, h)
    tracks, heights = {}, []
    for (actor, k), ob in sorted(groups.items()):
        if len(ob) < 3:
            continue
        X, err = triangulate(cams, {s: (x + w / 2, y + head_frac * h) for s, (x, y, w, h) in ob.items()})
        if err < 60:
            tracks.setdefault(f"nearrange-actor-{actor}", []).append((k / 30 - delta_v, X, err))
            top, e1 = triangulate(cams, {s: (x + w / 2, y) for s, (x, y, w, h) in ob.items()})
            bot, e2 = triangulate(cams, {s: (x + w / 2, y + h) for s, (x, y, w, h) in ob.items()})
            if max(e1, e2) < 80:
                heights.append(top[2] - bot[2])
    out = {}
    for a, lst in tracks.items():
        t = np.array([x[0] for x in lst])
        out[a] = (t, np.array([x[1] for x in lst]), float(np.median([x[2] for x in lst])))
    return out, np.array(heights)


INITS = [np.array(v) for v in ([0, 0, -0.3], [0.4, 0, -0.3], [-0.4, 0, -0.3], [0, 0.4, -0.3], [0, -0.4, -0.3])]


def interp_track(t_query, t, X, max_gap=0.2):
    j = np.searchsorted(t, t_query)
    ok = (j > 0) & (j < len(t))
    jj = np.clip(j, 1, len(t) - 1)
    gap = np.minimum(np.abs(t[jj] - t_query), np.abs(t[jj - 1] - t_query))
    ok &= gap < max_gap
    P = np.stack([np.interp(t_query, t, X[:, k]) for k in range(3)], 1)
    return P, ok


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "SC-1044"
    src = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), seq)
    db = sqlite3.connect(DB)
    cams, norm, delta_v = load_cameras(seq, db)
    vis, heights = visual_tracks(db, seq, cams, delta_v)
    for a, (t, X, e) in vis.items():
        print(f"visual {a}: {len(t)} frames, t=[{t.min():.1f},{t.max():.1f}]s (video-local), "
              f"reprojection {e:.1f}px", flush=True)
    h_units = float(np.percentile(heights, 85))
    mpu = H_ASSUMED / h_units
    print(f"standing height {h_units:.3f} units (85th pct of {len(heights)} frames) -> "
          f"{mpu:.3f} m/unit, ring radius {mpu:.2f} m", flush=True)

    t0 = time.time()
    A = AudioScene(src, cams)
    dur_a = min(len(b) for b in A.bins) / A.fs
    obs = A.observations(0, dur_a, hop=0.2)
    D = np.array([o[2] for o in obs])                      # (n, 8) lavalier->ear delays, s
    G = np.array([o[3] for o in obs]) > 4.0
    ITD = np.array([o[4] for o in obs])                    # (n, 8) right-minus-left ear delay, s
    GI = np.array([o[5] for o in obs]) > 4.0
    ta = np.array([o[1] for o in obs])
    who = np.array([o[0] for o in obs])
    M = A.Cm.shape[0]
    print(f"audio: {dur_a:.1f}s, {len(obs)} voiced windows ({time.time() - t0:.0f}s); "
          f"ITD usable per recorder {GI.sum(0).tolist()}", flush=True)

    def predict_pos(off):
        P = np.full((len(obs), 3), np.nan)
        for a, (tv, Xv, _) in vis.items():
            m = who == a
            if m.any():
                Pa, ok = interp_track(ta[m] - off, tv, Xv)
                Pa[~ok] = np.nan
                P[m] = Pa
        return P

    def itd_fit(P, mask=None):
        """Per recorder: itd = b_m * cos(theta_m) + c_m, trimmed LS. Returns b, c, median |res| (ms), n."""
        ok_w = np.isfinite(P[:, 0]) & (np.ones(len(P), bool) if mask is None else mask)
        b, c, res = np.zeros(M), np.zeros(M), []
        for m in range(M):
            w = np.nonzero(ok_w & GI[:, m])[0]
            if len(w) < 20:
                return None, None, np.inf, 0
            v = P[w] - A.Cm[m]
            cs = (v @ A.Rm[m]) / np.linalg.norm(v, axis=1)
            X = np.stack([cs, np.ones_like(cs)], 1)
            y = ITD[w, m]
            keep = np.ones(len(w), bool)
            for _ in range(3):
                p, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                r = (X @ p - y) * 1e3
                keep = np.abs(r) < max(0.05, 3 * np.median(np.abs(r[keep])))
            b[m], c[m] = p
            res.append(np.abs(r[keep]))
        res = np.concatenate(res)
        return b, c, float(np.median(res)), len(res)

    # ---- sync: timestamp_delta is the global time of each file's first sample (exact for video:
    # it equals the first frame timestamp), so audio_t = video_t + (delta_video - delta_audio).
    delta_a = db.execute("SELECT timestamp_delta FROM sensor_annots WHERE sequence_name=? AND sensor_name='binaural-1'",
                         (seq,)).fetchone()[0]
    offset = float(delta_v - delta_a)                    # video_t = audio_t - offset
    scan = np.arange(offset - 20, offset + 20.01, 0.5)
    ci = np.array([itd_fit(predict_pos(o))[2] for o in scan])
    print(f"sync from metadata: video_t = audio_t - ({offset:+.2f}) s;  ITD-cost minimum within +-20 s at "
          f"{scan[int(np.argmin(ci))]:+.1f} s ({ci.min() * 1e3:.0f} us vs {ci[np.argmin(np.abs(scan - offset))] * 1e3:.0f} us "
          f"at the metadata offset)", flush=True)

    # ---- calibrate on even seconds: ITD gains; latency per recorder (delta_m) and per lavalier (eps_a)
    P = predict_pos(offset)
    even = (ta.astype(int) % 2 == 0)
    bI, cI, rI, nI = itd_fit(P, even)
    actors = sorted(set(who.tolist()))

    def range_fit(dz, mask):
        dist = np.linalg.norm(P[:, None, :] + np.array([0, 0, dz]) - A.Cm[None], axis=2)
        sel = G & np.isfinite(dist) & mask[:, None]
        w, m = np.nonzero(sel)
        X = np.zeros((len(w), M + len(actors) - 1))
        X[np.arange(len(w)), m] = 1.0
        for k, a in enumerate(actors[1:]):
            X[who[w] == a, M + k] = 1.0
        y = D[w, m] - mpu * dist[w, m] / C_SOUND
        keep = np.ones(len(w), bool)
        for _ in range(3):
            p, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            r = np.abs(X @ p - y) * 1e3
            keep = r < max(0.3, 3 * np.median(r[keep]))
        eps = {actors[0]: 0.0, **{a: float(p[M + k]) for k, a in enumerate(actors[1:])}}
        return p[:M], eps, float(np.median(r))
    best = None
    for dz in np.linspace(-0.08, 0.08, 17):
        dl, eps, rr = range_fit(dz, even)
        if best is None or rr < best[3]:
            best = (dl, eps, dz, rr)
    dl, eps, dz, rr = best
    print(f"calibration (even seconds): ITD gain b [us] {np.round(bI * 1e6, 0)}, ITD residual {rI * 1e3:.0f} us; "
          f"range residual {rr:.3f} ms, source dz {dz * mpu * 100:+.1f} cm, per-recorder latency [ms] "
          f"{np.round(dl * 1e3, 2)}, per-lavalier latency [ms] {({k: round(v * 1e3, 2) for k, v in eps.items()})}",
          flush=True)

    # ---- held-out (odd seconds): audio-only positions vs video
    e3, exy = [], []
    for w in np.nonzero(~even & np.isfinite(P[:, 0]) & (G.sum(1) >= 5))[0]:
        X, r = None, np.inf
        for x0 in INITS:                                     # audio-only starts (no video prior)
            Xi, ri = solve_window(D[w] - eps[who[w]], G[w], A.Cm, mpu, dl, X0=x0, itd=ITD[w] - cI, gi=GI[w],
                                  Rm=A.Rm, b=bI)
            if ri < r:
                X, r = Xi, ri
        if r < 0.6:
            Pv = P[w] + np.array([0, 0, dz])
            e3.append(np.linalg.norm(X - Pv) * mpu * 100)
            exy.append(np.linalg.norm((X - Pv)[:2]) * mpu * 100)
    if e3:
        print(f"held-out ({len(e3)} windows): audio-vs-video error median 3D {np.median(e3):.1f} cm, "
              f"horizontal {np.median(exy):.1f} cm, 90th pct 3D {np.percentile(e3, 90):.1f} cm", flush=True)
    else:
        e3, exy = [np.nan], [np.nan]
        print("held-out: no window passed the residual gate", flush=True)
    dl = dl.tolist()

    out = {"seq": seq, "audio_offset_s": offset, "metres_per_unit": float(mpu), "latency_s": dl,
           "itd_gain_s": bI.tolist(), "itd_bias_s": cI.tolist(), "height_units": h_units,
           "lavalier_latency_s": eps, "sync_source": "metadata timestamp_delta (video - audio)",
           "source_dz_units": float(dz), "delta_video": delta_v, "norm": norm,
           "heldout_err_cm": {"median_3d": float(np.median(e3)), "median_xy": float(np.median(exy)),
                              "p90_3d": float(np.percentile(e3, 90)), "n": len(e3)},
           "visual": {a: {"t": t.tolist(), "pos": X.tolist()} for a, (t, X, _) in vis.items()}}
    json.dump(out, open(os.path.join(src, "sync.json"), "w"))
    print("SYNC_DONE")


if __name__ == "__main__":
    main()

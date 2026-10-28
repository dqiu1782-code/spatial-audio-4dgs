"""End-to-end check on a simulated capture: clap sync -> self-calibration -> audio-only localisation.

  python tests/test_selfcal.py examples/rig_advanced_binaural.json [--no-lav]
Calibrates on the first half of the take, evaluates localisation on the second half (held out).
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rig import load_rig
from selfcal import calibrate, extract, localize, nominal_params, track_fn
from sim import cached_simulation
from sync import sync_recordings

rig_path = sys.argv[1]
use_lav = "--no-lav" not in sys.argv
rig = load_rig(rig_path)
t0 = time.time()
S = cached_simulation(rig, rig_path)
print(f"capture ready in {time.time() - t0:.0f}s")
fs, truth = S["fs"], S["truth"]
lavs = S["lavs"] if use_lav else {}
tracks_true = truth["talkers"]
rng = np.random.default_rng(5)
tracks_video = [dict(name=t["name"], t=t["t"], pos=t["pos"] + rng.normal(0, 0.03, t["pos"].shape)) for t in tracks_true]

# ---- clap sync
recs = {m.id: S["mics"][m.id][:, 0] for m in rig.mics}
pos = {m.id: rig.channel_positions_world(m)[0] for m in rig.mics}
if use_lav:
    for k, v in lavs.items():
        recs["lav:" + k] = v
        pos["lav:" + k] = track_fn(next(t for t in tracks_true if t["name"] == k))(np.array([truth["clap_times"][0]]))[0]
ref = "lav:" + next(iter(lavs)) if use_lav else rig.mics[0].id
off, _ = sync_recordings(recs, fs, np.array(truth["clap_pos"]), pos, ref=ref)
tm = truth["mics"]
ref_true = tm[ref]["latency"] if use_lav else tm[ref]["latency"] - tm[ref]["start"]
err = [abs((off[m.id]) - ((tm[m.id]["latency"] - tm[m.id]["start"]) - ref_true)) * 1e3 for m in rig.mics]
print(f"clap sync: median |offset error| {np.median(err):.2f} ms, max {np.max(err):.2f} ms "
      f"(recorder start times spread {np.ptp([tm[m.id]['start'] for m in rig.mics]):.2f} s)")

# ---- observations, calibration on first half
t0 = time.time()
obs = extract(rig, S["mics"], lavs, {m.id: off[m.id] for m in rig.mics}, fs)
half = truth["dur"] / 2
print(f"{len(obs['t'])} attributable windows ({time.time() - t0:.0f}s)")
cal = calibrate(rig, obs, tracks_video, fit_mask=obs["t"] < half)
nom = nominal_params(rig)
for i, m in enumerate(rig.mics[:4]):
    p, tr = cal["params"][i], tm[m.id]
    yaw_err = (p["yaw"] - tr["yaw"] + 180) % 360 - 180
    off_err = np.linalg.norm(m.offset + np.array([p["dx"], p["dy"], p["dz"]]) - tr["offset"]) * 100
    print(f"  {m.id}: axis error {yaw_err:+.0f} deg (true yaw {tr['yaw']:.0f}), offset error {off_err:.1f} cm, "
          f"drift {p['drift'] * 1e6:+.0f} ppm (true {-tr['drift'] * 1e6:+.0f})")
if not use_lav:
    att = cal["actor"]
    print(f"  attributed windows: {np.mean(att >= 0) * 100:.0f}% to a talker, {np.mean(att < 0) * 100:.0f}% rejected")

# ---- held-out localisation: nominal geometry vs self-calibrated
test = obs["t"] >= half
if not use_lav:
    obs["actor"] = cal["actor"]
truthfn = {t["name"]: track_fn(t) for t in tracks_true}
for label, params, ll in (("nominal (clap sync only)", nom, {k: 0.0 for k in cal["lav_latency"]}),
                          ("self-calibrated", cal["params"], cal["lav_latency"])):
    idx, X, rms = localize(rig, obs, params, lav_latency=ll if use_lav else None,
                           mask=test & (obs["actor"] >= 0))
    names = [t["name"] for t in tracks_true]
    Xt = np.array([truthfn[names[obs["actor"][w]]](np.array([obs["t"][w]]))[0] for w in idx])
    e = np.linalg.norm(X - Xt, axis=1) * 100
    exy = np.linalg.norm((X - Xt)[:, :2], axis=1) * 100
    print(f"held-out localisation, {label}: median 3D {np.median(e):.1f} cm, horizontal {np.median(exy):.1f} cm, "
          f"height {np.median(np.abs((X - Xt)[:, 2])) * 100:.1f} cm, 90th pct {np.percentile(e, 90):.1f} cm (n={len(e)})")

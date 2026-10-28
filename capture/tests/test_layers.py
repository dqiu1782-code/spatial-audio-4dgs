"""Voice/ambience layering on a simulated capture (lavalier references vs beamformed references).

  python tests/test_layers.py examples/rig_advanced_binaural.json [--no-lav]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layers import activity_gate, beamform_refs, ctf_separate, leakage_db
from rig import load_rig
from sim import cached_simulation
from selfcal import Channels, calibrate, extract, track_fn
from sync import sync_recordings

rig_path = sys.argv[1]
use_lav = "--no-lav" not in sys.argv
rig = load_rig(rig_path)
S = cached_simulation(rig, rig_path)
fs, truth = S["fs"], S["truth"]
rng = np.random.default_rng(5)
tracks = [dict(name=t["name"], t=t["t"], pos=t["pos"] + rng.normal(0, 0.03, t["pos"].shape)) for t in truth["talkers"]]
recs = {m.id: S["mics"][m.id][:, 0] for m in rig.mics}
pos = {m.id: rig.channel_positions_world(m)[0] for m in rig.mics}
lavs = S["lavs"] if use_lav else {}
for k, v in lavs.items():
    recs["lav:" + k] = v
    pos["lav:" + k] = track_fn(next(t for t in truth["talkers"] if t["name"] == k))(np.array([truth["clap_times"][0]]))[0]
ref = "lav:" + next(iter(lavs)) if use_lav else rig.mics[0].id
off, _ = sync_recordings(recs, fs, np.array(truth["clap_pos"]), pos, ref=ref)
moff = {m.id: off[m.id] for m in rig.mics}

t0 = time.time()
if use_lav:
    n = min(len(v) for v in lavs.values())
    refs = np.stack([lavs[k][:n] for k in sorted(lavs)], 1)
    label = "lavalier references"
else:
    obs = extract(rig, S["mics"], {}, moff, fs)
    cal = calibrate(rig, obs, tracks, verbose=False)
    refs = beamform_refs(rig, S["mics"], moff, cal["params"], tracks, fs, Channels(rig).positions(cal["params"]))
    if "--no-gate" not in sys.argv:
        # obs["t"] is on the reference clock; gate each beam by the windows attributed to its talker
        refs = refs * activity_gate(obs["t"], cal["actor"], len(tracks), len(refs), fs)
    label = "beamformed references (no lavaliers)" + ("" if "--no-gate" in sys.argv else ", activity-gated")
print(f"references: {label} ({time.time() - t0:.0f}s)")

rows = []
for m in rig.mics[:4]:
    Y = S["mics"][m.id].astype(float)
    s = int(round(moff[m.id] * fs))                    # put the mic on the reference clock
    Ya = Y[s:] if s >= 0 else np.pad(Y, ((-s, 0), (0, 0)))
    n = min(len(Ya), len(refs))
    Ya, R = Ya[:n], refs[:n]
    Vt = S["voice"][m.id][s:s + n].astype(float) if s >= 0 else np.pad(S["voice"][m.id], ((-s, 0), (0, 0)))[:n]
    Bt = S["background"][m.id][s:s + n].astype(float) if s >= 0 else np.pad(S["background"][m.id], ((-s, 0), (0, 0)))[:n]
    t1 = time.time()
    v, amb = ctf_separate(Ya, R, fs)
    base = leakage_db(Ya, Bt, Vt)                        # "no separation": ambience = raw recording
    sep = leakage_db(amb, Bt, Vt)
    corr = np.corrcoef(amb.ravel(), Bt.ravel())[0, 1]
    if not use_lav:
        # hybrid bed: raw recording between turns, CTF residual during speech
        act = np.clip(activity_gate(obs["t"], cal["actor"], len(tracks), n, fs).sum(1), 0, 1)[:, None]
        hyb = Ya * (1 - act) + amb * act
        print(f"    hybrid bed: {leakage_db(hyb, Bt, Vt):+.1f} dB, corr {np.corrcoef(hyb.ravel(), Bt.ravel())[0, 1]:.2f}, "
              f"speech-active {act.mean():.0%} of time")
    rows.append((base, sep, corr))
    print(f"  {m.id}: voice left in the ambience layer {base:+.1f} dB (raw) -> {sep:+.1f} dB (layered); "
          f"ambience vs true background corr {corr:.2f}  ({time.time() - t1:.0f}s)")
r = np.array(rows)
print(f"MEAN: voice suppressed by {r[:, 0].mean() - r[:, 1].mean():.1f} dB in the ambience layer; "
      f"ambience/background correlation {r[:, 2].mean():.2f}")

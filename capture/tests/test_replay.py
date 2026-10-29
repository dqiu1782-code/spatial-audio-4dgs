"""Self-calibration and layering on real data: Meta Replay scene (8 DSLRs + 8 binaural + lavaliers).

  python tests/test_replay.py SC-1044
Needs ../replay/replay_common.py, E:\\replay\\<seq>\\ (flac + sync.json from replay_sync.py).
Calibrates on the first half, evaluates on the second half against the video-triangulated heads.
Writes capture/out/<seq>_selfcal.json and voice/ambience wav files.
"""
import json
import os
import sqlite3
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "..", "replay"))       # replay_common
from layers import ctf_separate                              # noqa: E402
from replay_common import DB, SENSORS, load_cameras          # noqa: E402
from rig import Camera, Mic, Rig                             # noqa: E402
from selfcal import Channels, _model, calibrate, extract, localize, nominal_params, track_fn  # noqa: E402

FS = 16000
seq = sys.argv[1] if len(sys.argv) > 1 else "SC-1044"
src = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), seq)
out_dir = os.path.join(PKG, "out")
os.makedirs(out_dir, exist_ok=True)
S = json.load(open(os.path.join(src, "sync.json")))
mpu = S["metres_per_unit"]
db = sqlite3.connect(DB)
cams, _, delta_v = load_cameras(seq, db)
deltas = dict(db.execute("SELECT sensor_name, timestamp_delta FROM sensor_annots WHERE sequence_name=?", (seq,)).fetchall())

# ---- rig: camera poses from the dataset, mic placement unknown (to be calibrated)
cameras = []
for k, s in enumerate(SENSORS):
    c = Camera(s, k + 1, "dslr", np.asarray(cams[s]["Cn"]) * mpu, np.asarray(cams[s]["Rn"]), f"{s}.mp4", 60.0)
    c.mics.append(Mic(f"binaural-{k + 1}", "recorder", "binaural", 0.17, np.zeros(3), 0.0, 0.0,
                      f"binaural-{k + 1}.flac", [0, 1], latency_sd=0.05, offset_sd=float(os.environ.get("OFFSET_SD", "0.15")), drift_sd=50e-6))
    cameras.append(c)
lav_files = sorted(f for f in os.listdir(src) if f.startswith("nearrange") and f.endswith(".flac"))
rig = Rig(f"replay_{seq}", "advanced", cameras, [{"actor": f[:-5]} for f in lav_files], {}, {"method": "metadata"}, {}, {})


def load(f):
    x, fs = sf.read(os.path.join(src, f), always_2d=True)
    return resample_poly(x, FS, fs, axis=0).astype(np.float32)


t0 = time.time()
mics = {m.id: load(m.audio) for m in rig.mics}
lavs = {f[:-5]: load(f).mean(1) for f in lav_files}
names = list(lavs)
ref = names[0]
print(f"audio loaded ({time.time() - t0:.0f}s): {len(mics)} binaural, lavaliers {names}; "
      f"duration {min(len(v) for v in mics.values()) / FS:.1f}s", flush=True)
missing = [k for k in list(mics) + names if k not in deltas]
if missing:
    print("no timestamp_delta for", missing, "; sensors:", sorted(deltas))
    sys.exit(1)
# timestamp_delta = global time of the file's first sample -> t_rec = t_phys + (delta_ref - delta_file)
off = {k: deltas[ref] - deltas[k] for k in mics}
lav_off = {k: deltas[ref] - deltas[k] for k in names}
print("offsets vs " + ref + " [s]: " + ", ".join(f"{k} {v:+.3f}" for k, v in {**off, **lav_off}.items()), flush=True)
for k, v in lav_off.items():                 # put every lavalier on the reference clock
    s = int(round(v * FS))
    lavs[k] = lavs[k][s:] if s >= 0 else np.pad(lavs[k], (-s, 0))

# ---- video tracks (head points) on the reference clock, metres
tracks = []
for a in names:
    v = S["visual"][a]
    tracks.append(dict(name=a, t=np.asarray(v["t"]) + delta_v - deltas[ref], pos=np.asarray(v["pos"]) * mpu))
t_lo = max(min(tr["t"].min() for tr in tracks), 5.0)
t_hi = min(tr["t"].max() for tr in tracks)
print(f"video tracks cover {t_lo:.1f}-{t_hi:.1f}s on the reference clock", flush=True)

# ---- observations
t0 = time.time()
obs = extract(rig, mics, lavs, off, FS, max_lag=0.1, t_range=(t_lo, t_hi))
mid = 0.5 * (t_lo + t_hi)
FIT = (obs["t"] < mid) if os.environ.get("SWAP") != "1" else (obs["t"] >= mid)
TEST = ~FIT
print(f"{len(obs['t'])} lavalier-attributed windows ({time.time() - t0:.0f}s); per actor "
      f"{[int((obs['actor'] == a).sum()) for a in range(len(names))]}", flush=True)

cal = calibrate(rig, obs, tracks, iters=6, fit_mask=FIT)
nom = nominal_params(rig)
nom_ll = {k: 0.0 for k in names}
ch = Channels(rig)
for i, m in enumerate(rig.mics):
    p = cal["params"][i]
    print(f"  {m.id}: latency {p['lat'] * 1e3:+7.2f} ms, drift {p['drift'] * 1e6:+5.0f} ppm, ear axis yaw "
          f"{p['yaw']:+6.0f} deg, pitch {p['pitch']:+4.0f}, offset (cam x,y,z) "
          f"[{p['dx'] * 100:+.0f}, {p['dy'] * 100:+.0f}, {p['dz'] * 100:+.0f}] cm", flush=True)
print(f"  lavalier latency [ms]: {({k: round(v * 1e3, 2) for k, v in cal['lav_latency'].items()})}", flush=True)


def heldout_prediction(params, lav_latency):
    """Median |measured - predicted| delay (ms) on held-out windows with the video positions."""
    P = ch.positions(params)
    fns = [track_fn(tr) for tr in tracks]
    errs = []
    for w in np.nonzero(TEST & (obs["actor"] >= 0))[0]:
        g = obs["q"][w] > 4.0
        if g.sum() < 5:
            continue
        a = obs["actor"][w]
        X = fns[a](np.array([obs["t"][w]]))
        r = obs["tau"][w] - _model(P, X, np.array([obs["t"][w]]), params, ch.dev)[0] - lav_latency[names[a]]
        errs.append(np.median(np.abs(r[g])) * 1e3)
    return float(np.median(errs)), len(errs)


# nominal reference: lavalier latency fitted with the nominal geometry (otherwise it is unfairly bad)
P0 = ch.positions(nom)
fns = [track_fn(tr) for tr in tracks]
for a, k in enumerate(names):
    w = np.nonzero((obs["actor"] == a) & FIT)[0]
    r = [obs["tau"][i] - _model(P0, fns[a](np.array([obs["t"][i]])), np.array([obs["t"][i]]), nom, ch.dev)[0]
         for i in w]
    nom_ll[k] = float(np.median(np.concatenate(r))) if r else 0.0
res = {}
for label, params, ll in (("nominal", nom, nom_ll), ("self-calibrated", cal["params"], cal["lav_latency"])):
    pe, n_pe = heldout_prediction(params, ll)
    centre = np.mean([tr["pos"].mean(0) for tr in tracks], 0)
    inits = [centre + np.array(d) for d in ([0, 0, 0], [0.8, 0, 0], [-0.8, 0, 0], [0, 0.8, 0], [0, -0.8, 0])]
    idx, X, rms = localize(rig, obs, params, lav_latency=ll, mask=TEST & (obs["actor"] >= 0),
                           inits=inits)
    ok = rms < 1.0
    Xt = np.array([fns[obs["actor"][w]](np.array([obs["t"][w]]))[0] for w in idx])
    e = np.linalg.norm(X - Xt, axis=1)[ok] * 100
    exy = np.linalg.norm((X - Xt)[:, :2], axis=1)[ok] * 100
    res[label] = dict(pred_ms=pe, n_pred=n_pe, n_loc=int(ok.sum()), n_try=len(idx),
                      loc_3d_cm=float(np.median(e)) if len(e) else None,
                      loc_xy_cm=float(np.median(exy)) if len(e) else None,
                      loc_p90_cm=float(np.percentile(e, 90)) if len(e) else None)
    print(f"held-out, {label}: delay prediction error {pe:.3f} ms (n={n_pe}); audio-only localisation "
          + (f"median 3D {np.median(e):.1f} cm, horizontal {np.median(exy):.1f} cm, p90 {np.percentile(e, 90):.1f} cm, height bias {np.median((X - Xt)[ok, 2]) * 100:+.1f} cm "
             f"({ok.sum()}/{len(idx)} windows with rms < 1 ms)" if len(e) else f"no window converged ({len(idx)} tried)"),
          flush=True)

# ---- layering on one binaural recorder: voice estimate from the lavaliers, residual = ambience
m = rig.mics[0]
s = int(round(off[m.id] * FS))
Y = mics[m.id].astype(float)
Ya = Y[s:] if s >= 0 else np.pad(Y, ((-s, 0), (0, 0)))
i0 = int(t_lo * FS)                         # 40 s excerpt (memory: the CTF regressors grow with length)
n = min(len(Ya), min(len(v) for v in lavs.values()), i0 + 40 * FS) - i0
Ya = Ya[i0:]
lavs = {k: v[i0:] for k, v in lavs.items()}
R = np.stack([lavs[k][:n] for k in names], 1).astype(float)
t0 = time.time()
v, amb = ctf_separate(Ya[:n], R, FS)
hop = FS // 10
nb = n // hop
E = np.array([np.mean(lavs[k][: nb * hop].reshape(nb, hop) ** 2, 1) for k in names])
speech = (E.max(0) > np.percentile(E.max(0), 60)).repeat(hop)
gap = (E.max(0) < np.percentile(E.max(0), 20)).repeat(hop)
nn = len(speech)


def db(x, mask):
    return 10 * np.log10(np.mean(x[:nn][mask] ** 2) + 1e-20)


drop_speech = db(Ya[:n], speech) - db(amb, speech)
drop_gap = db(Ya[:n], gap) - db(amb, gap)
print(f"layering {m.id} ({time.time() - t0:.0f}s): ambience layer is {drop_speech:.1f} dB below the raw recording "
      f"while someone talks, {drop_gap:.1f} dB in pauses (pauses should stay ~0 dB)", flush=True)
for name, x in (("raw", Ya[:n]), ("voice", v), ("ambience", amb)):
    sf.write(os.path.join(out_dir, f"{seq}_{m.id}_{name}.wav"), (x / (np.abs(Ya[:n]).max() + 1e-9) * 0.9).astype(np.float32), FS)
res["layering"] = dict(mic=m.id, drop_speech_db=float(drop_speech), drop_pause_db=float(drop_gap))
res["params"] = {mm.id: cal["params"][i] for i, mm in enumerate(rig.mics)}
res["lav_latency_s"] = cal["lav_latency"]
res["fit"] = dict(residual_ms=cal["residual_ms"], inlier_rate=cal["inlier_rate"])
json.dump(res, open(os.path.join(out_dir, f"{seq}_selfcal.json"), "w"), indent=1, default=float)
print("REPLAY_TEST_DONE")

"""Microphone self-calibration from talkers at known (video-tracked) positions.

Which sound is whose
--------------------
Windows of 0.25 s are kept only if they can be attributed to one talker:
  * with lavaliers: that talker's lavalier is >= dom_db louder than all others (solo speech) and is
    used as the reference; delays to every mic come from GCC-PHAT against it, so sound that is not
    coherent with this voice (AC, door, other talker, crew) produces no peak and is ignored;
  * without lavaliers: the loudest channel is the reference and the window is attributed to the
    talker whose (video) position best explains the measured delays; if none does, the window is
    labelled background (door knock, AC, ...) and excluded.
Per-channel peaks below q_min and fit outliers (> 3 robust sigma) are dropped as well.

Model
-----
tau_wk = |X_a(t_w) - p_k(theta_d)| / c + lat_d + drift_d * t_w + b_w
  p_k    channel position from the rig prior plus device corrections theta_d:
         mic offset (3, prior per device type), axis yaw/pitch (prior 30 deg; L/R flips tested)
  lat_d  residual latency after clap sync; drift_d clock drift (prior per device type)
  b_w    per-talker lavalier latency (lavaliers) or per-window emission time (no lavaliers)
Solved by alternating robust least squares: reference terms, then each device independently.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt

C = 343.0
PARAMS = ("lat", "drift", "dx", "dy", "dz", "yaw", "pitch")


def gcc(ref, sig, max_lag, fs, band=(200, 5000)):
    """Lag (s) of sig relative to ref, and peak sharpness."""
    n = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    X = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X *= ((f > band[0]) & (f < band[1])) / (np.abs(X) + 1e-12)
    cc = np.fft.irfft(X, n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(cc))
    kf = float(k)
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        kf = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return (kf - max_lag) / fs, cc[k] / (np.mean(np.abs(cc)) + 1e-12)


class Channels:
    """Flat list of (mic, channel index) with device indices."""

    def __init__(self, rig):
        self.rig = rig
        self.items = [(m, j) for m in rig.mics for j in range(len(m.channels))]
        self.dev = np.array([i for i, m in enumerate(rig.mics) for _ in m.channels])
        self.mics = rig.mics

    def positions(self, params):
        P = []
        for i, m in enumerate(self.mics):
            p = params[i]
            P.append(self.rig.channel_positions_world(
                m, offset=m.offset + np.array([p["dx"], p["dy"], p["dz"]]), yaw=p["yaw"], pitch=p["pitch"]))
        return np.concatenate(P)


def nominal_params(rig):
    return [dict(lat=0.0, drift=0.0, dx=0.0, dy=0.0, dz=0.0, yaw=m.axis_yaw, pitch=m.axis_pitch) for m in rig.mics]


def track_fn(track):
    t, P = np.asarray(track["t"]), np.asarray(track["pos"])
    return lambda q: np.stack([np.interp(q, t, P[:, j]) for j in range(3)], -1)


def extract(rig, mics, lavs, offsets, fs, win=0.25, hop=0.1, max_lag=0.03, dom_db=6.0, t_range=None):
    """Delay observations. offsets: {mic_id: s} from clap sync (t_rec = t_phys + offset)."""
    ch = Channels(rig)
    sos = butter(4, [150, min(6000, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    sig = {m.id: sosfiltfilt(sos, mics[m.id], axis=0) for m in rig.mics}
    lav = {k: sosfiltfilt(sos, v) for k, v in lavs.items()}
    W, L = int(win * fs), int(max_lag * fs)
    names = list(lav)
    o_max = max(max(offsets[m.id] for m in rig.mics), 0.0)
    o_min = min(min(offsets[m.id] for m in rig.mics), 0.0)
    n = min(len(v) for v in sig.values()) - W - 2 * L - int((o_max + 0.05) * fs)
    if names:
        n = min(n, min(len(v) for v in lav.values()) - W)
    t0, t1 = t_range or (0.0, n / fs)
    first = max(t0, 5.0, (L / fs) - o_min + 0.05)
    starts = np.arange(int(first * fs), int(min(t1, n / fs) * fs), int(hop * fs))
    if names:
        E = np.array([[np.mean(lav[a][i:i + W] ** 2) for i in starts] for a in names])
        floor = np.percentile(E, 40, axis=1)[:, None]
    tw, actor, tau, q, refk = [], [], [], [], []
    for j, i in enumerate(starts):
        if names:
            a = int(np.argmax(E[:, j]))
            others = E[:, j].sum() - E[a, j] + 1e-12
            if E[a, j] < floor[a, 0] or 10 * np.log10(E[a, j] / others) < dom_db:
                continue
            ref, rk = lav[names[a]][i:i + W], -1
        else:
            a = -1
            en = [np.mean(sig[m.id][int(i + offsets[m.id] * fs): int(i + offsets[m.id] * fs) + W, c] ** 2)
                  for m, c in ch.items]
            rk = int(np.argmax(en))
            m, c = ch.items[rk]
            s0 = int(i + offsets[m.id] * fs)
            ref = sig[m.id][s0:s0 + W, c]
        d, qq = [], []
        for m, c in ch.items:
            s0 = int(round(i + offsets[m.id] * fs)) - L
            frac = (i + offsets[m.id] * fs - L) - s0
            lag, qual = gcc(ref, sig[m.id][s0:s0 + W + 2 * L, c], 2 * L, fs)
            d.append(lag - (L + frac) / fs)      # segment starts frac samples before the ideal point
            qq.append(qual)
        tw.append((i + W / 2) / fs)
        actor.append(a)
        tau.append(d)
        q.append(qq)
        refk.append(rk)
    return dict(t=np.array(tw), actor=np.array(actor), tau=np.array(tau), q=np.array(q), ref=np.array(refk),
                names=names, channels=ch)


def _model(P, X, t, p_dev, dev):
    lat = np.array([p["lat"] for p in p_dev])[dev]
    drift = np.array([p["drift"] for p in p_dev])[dev]
    return np.linalg.norm(X[:, None, :] - P[None], axis=2) / C + lat[None] + drift[None] * t[:, None]


def attribute(obs, tracks, ch, params, thr_ms=0.5):
    """No-lavalier case: pick the talker whose position explains the delays (per-window offset free)."""
    P = ch.positions(params)
    best = np.full(len(obs["t"]), -1)
    fns = [track_fn(tr) for tr in tracks]
    good = obs["q"] > 4.0
    for w in range(len(obs["t"])):
        errs = []
        for a, fn in enumerate(fns):
            X = fn(np.array([obs["t"][w]]))
            r = obs["tau"][w] - _model(P, X, np.array([obs["t"][w]]), params, ch.dev)[0]
            r = r[good[w]] - np.median(r[good[w]]) if good[w].sum() >= 4 else np.array([np.inf])
            errs.append(np.median(np.abs(r)) * 1e3)
        a = int(np.argmin(errs))
        best[w] = a if errs[a] < thr_ms else -1
    return best


def calibrate(rig, obs, tracks, iters=5, q_min=4.0, prior_yaw=30.0, fit_mask=None, verbose=True):
    """Returns per-device params, per-talker/window reference terms and diagnostics."""
    ch = obs["channels"]
    params = nominal_params(rig)
    fns = [track_fn(tr) for tr in tracks]
    lav_mode = len(obs["names"]) > 0
    actor = obs["actor"].copy()
    if not lav_mode:
        actor = attribute(obs, tracks, ch, params, thr_ms=2.0)
    fit_mask = np.ones(len(actor), bool) if fit_mask is None else fit_mask

    def select(actor):
        use = (actor >= 0) & fit_mask
        t, act = obs["t"][use], actor[use]
        X = np.zeros((len(t), 3))
        for a, fn in enumerate(fns):
            X[act == a] = fn(t[act == a])
        return use, t, obs["tau"][use], act, obs["q"][use] > q_min, X
    use, t, tau, act, good, X = select(actor)
    inl = good.copy()
    for it in range(iters):
        if not lav_mode and it > 0:                    # re-attribute with the current calibration, tighter gate
            actor = attribute(obs, tracks, ch, params, thr_ms=max(0.6, 2.0 / (it + 1)))
            use, t, tau, act, good, X = select(actor)
            inl = good.copy()
        b = np.zeros(len(t))
        P = ch.positions(params)
        r0 = tau - _model(P, X, t, params, ch.dev)
        if lav_mode:                                   # one latency per lavalier
            for a in np.unique(act):
                m = act == a
                b[m] = np.median(r0[m][inl[m]])
        else:                                          # free emission time per window
            b = np.array([np.median(r0[w][inl[w]]) if inl[w].sum() else 0.0 for w in range(len(t))])
        if it == 0:                                    # coarse start: each device's median residual latency
            for d in range(len(ch.mics)):
                cols = np.nonzero(ch.dev == d)[0]
                rr = (r0[:, cols] - b[:, None])[inl[:, cols]]
                if len(rr):
                    params[d]["lat"] += float(np.median(rr))
        for d, mic in enumerate(ch.mics):
            cols = np.nonzero(ch.dev == d)[0]
            p0 = params[d]
            cands = [p0["yaw"]] + ([p0["yaw"] + 180.0] if it == 0 and len(cols) > 1 else [])
            best = None
            for y0 in cands:
                x0 = np.array([p0["lat"], p0["drift"] * 1e6, p0["dx"], p0["dy"], p0["dz"], y0, p0["pitch"]])
                yaw_prior = y0

                def res(x, cols=cols, mic=mic, yaw_prior=yaw_prior):
                    pd = dict(zip(PARAMS, [x[0], x[1] * 1e-6, *x[2:]]))
                    Pd = rig.channel_positions_world(mic, offset=mic.offset + x[2:5], yaw=x[5], pitch=x[6])
                    mod = np.linalg.norm(X[:, None, :] - Pd[None], axis=2) / C + pd["lat"] + pd["drift"] * t[:, None]
                    r = ((tau[:, cols] - b[:, None] - mod) * 1e3)[inl[:, cols]]
                    so = mic.offset_sd
                    pri = [x[2] / so, x[3] / so, x[4] / so,
                           (x[5] - yaw_prior) / prior_yaw, (x[6] - mic.axis_pitch) / prior_yaw,
                           x[1] * 1e-6 / mic.drift_sd]
                    return np.concatenate([r, 0.3 * np.array(pri)])
                sol = least_squares(res, x0, loss="soft_l1", f_scale=0.15)
                if best is None or sol.cost < best.cost:
                    best = sol
            x = best.x
            params[d] = dict(zip(PARAMS, [x[0], x[1] * 1e-6, *x[2:]]))
        P = ch.positions(params)
        r = (tau - b[:, None] - _model(P, X, t, params, ch.dev)) * 1e3
        mad = np.median(np.abs(r[good])) * 1.4826 + 1e-3
        inl = good & (np.abs(r) < max(3 * mad, 0.15))
        if verbose:
            print(f"  iter {it + 1}: median |res| {np.median(np.abs(r[inl])):.3f} ms, inliers {inl.mean() * 100:.0f}%, "
                  f"windows {len(t)}", flush=True)
    return dict(params=params, b=b, actor=actor, used=use, inlier_rate=float(inl.mean()),
                residual_ms=float(np.median(np.abs(r[inl]))),
                lav_latency={obs["names"][a]: float(np.median(b[act == a])) for a in np.unique(act)} if lav_mode else {})


def localize(rig, obs, params, lav_latency=None, mask=None, q_min=4.0, inits=None):
    """Audio-only talker positions per window given calibration. Returns (idx, X, rms_ms)."""
    ch = obs["channels"]
    P = ch.positions(params)
    lat = np.array([p["lat"] for p in params])[ch.dev]
    drift = np.array([p["drift"] for p in params])[ch.dev]
    inits = inits or [np.array(v) for v in ([0, 0, 1.4], [0.6, 0, 1.4], [-0.6, 0, 1.4], [0, 0.6, 1.4], [0, -0.6, 1.4])]
    lav_mode = bool(lav_latency)
    out_i, out_X, out_r = [], [], []
    for w in np.nonzero(np.ones(len(obs["t"]), bool) if mask is None else mask)[0]:
        g = obs["q"][w] > q_min
        if g.sum() < 5:
            continue
        y = obs["tau"][w] - lat - drift * obs["t"][w]
        if lav_mode:
            if obs["actor"][w] < 0:
                continue
            y = y - lav_latency[obs["names"][obs["actor"][w]]]

        def res(x):
            r = np.linalg.norm(x[:3][None] - P, axis=1) / C - y + (0 if lav_mode else x[3])
            return r[g] * 1e3
        best = None
        for x0 in inits:
            x0 = x0 if lav_mode else np.append(x0, np.median(y[g] - np.linalg.norm(x0[None] - P, axis=1)[g] / C))
            s = least_squares(res, x0, loss="soft_l1", f_scale=0.15)
            if best is None or s.cost < best.cost:
                best = s
        rr = np.abs(res(best.x))
        out_i.append(w)
        out_X.append(best.x[:3])
        out_r.append(float(np.median(rr)))
    return np.array(out_i, int), np.array(out_X), np.array(out_r)

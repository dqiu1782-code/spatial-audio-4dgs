"""Shared helpers for the Replay pipeline: cameras, world normalisation, audio delays, 3D fits."""
import math
import os
import sqlite3

import numpy as np

DB = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "metadata", "metadata.sqlite")
SENSORS = [f"DSLR-{i}" for i in range(1, 9)]
C_SOUND = 343.0


# ---------------------------------------------------------------- cameras
def f32(blob, n):
    return np.frombuffer(blob, "<f4", count=n).astype(np.float64)


def p3d_to_cv(R, T):
    """PyTorch3D (x_cam = R^T x_w + T; +x left, +y up, +z fwd) -> OpenCV cam-to-world (R_c2w, C)."""
    F = np.diag([-1.0, -1.0, 1.0])
    Rc2w = (F @ R.T).T
    return Rc2w, -Rc2w @ (F @ T)


def ortho(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else -R


def c2w_gl(R, C):
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = R[:, 0], -R[:, 1], -R[:, 2], C
    return M


def load_cameras(seq, db=None):
    """Static DSLR cameras (pose/intrinsics averaged over the sequence) in a normalised world:
    ring centre at origin, mean camera up = +z, ring radius = 1. Returns (cams, norm, delta_video)."""
    db = db or sqlite3.connect(DB)
    delta = db.execute("SELECT timestamp_delta FROM sensor_annots WHERE sequence_name=? AND sensor_name='DSLR-1'",
                       (seq,)).fetchone()[0]
    cams = {}
    for s in SENSORS:
        rows = db.execute("SELECT _viewpoint_R, _viewpoint_T, _viewpoint_focal_length, _viewpoint_principal_point, "
                          "_image_size FROM frame_annots WHERE sequence_name=? AND sensor_name=? AND frame_number % 15 = 0",
                          (seq, s)).fetchall()
        Rs = np.array([f32(r[0], 9).reshape(3, 3) for r in rows])
        Ts = np.array([f32(r[1], 3) for r in rows])
        fl = np.array([f32(r[2], 2) for r in rows]).mean(0)
        pp = np.array([f32(r[3], 2) for r in rows]).mean(0)
        H0, W0 = np.frombuffer(rows[0][4], "<i4", count=2)
        R, C = p3d_to_cv(ortho(Rs.mean(0)), Ts.mean(0))
        sc = min(H0, W0) / 2.0
        K = dict(fx=fl[0] * sc, fy=fl[1] * sc, cx=-pp[0] * sc + W0 / 2, cy=-pp[1] * sc + H0 / 2, W=int(W0), H=int(H0))
        cams[s] = dict(R=R, C=C, K=K, jitter=float(np.linalg.norm(Ts - Ts.mean(0), axis=1).max()))
    Cs = np.array([cams[s]["C"] for s in SENSORS])
    c0 = Cs.mean(0)
    up = np.mean([-cams[s]["R"][:, 1] for s in SENSORS], 0)
    up /= np.linalg.norm(up)
    e1 = (Cs[0] - c0) - np.dot(Cs[0] - c0, up) * up
    e1 /= np.linalg.norm(e1)
    Rw = np.stack([e1, np.cross(up, e1), up])
    radius = float(np.mean(np.linalg.norm((Cs - c0) - ((Cs - c0) @ up)[:, None] * up, axis=1)))
    k = 1.0 / radius
    for s in SENSORS:
        cams[s]["Rn"] = Rw @ cams[s]["R"]
        cams[s]["Cn"] = k * Rw @ (cams[s]["C"] - c0)
    return cams, {"centre": c0.tolist(), "R": Rw.tolist(), "scale": k}, delta


def triangulate(cams, obs):
    """obs: {sensor: (u, v)} in original pixels -> least-squares 3D point (normalised world) and RMS px error."""
    A, rays = [], []
    for s, (u, v) in obs.items():
        c = cams[s]
        K = c["K"]
        d = c["Rn"] @ np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
        d /= np.linalg.norm(d)
        rays.append((c["Cn"], d))
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        M += P
        b += P @ C
    X = np.linalg.solve(M, b)
    err = []
    for s, (u, v) in obs.items():
        c = cams[s]
        q = c["Rn"].T @ (X - c["Cn"])
        K = c["K"]
        err.append(np.hypot(K["fx"] * q[0] / q[2] + K["cx"] - u, K["fy"] * q[1] / q[2] + K["cy"] - v))
    return X, float(np.sqrt(np.mean(np.square(err))))


# ---------------------------------------------------------------- audio
def gcc_phat(ref, sig, max_lag, fs, band=(250, 4000)):
    """Delay of sig relative to ref (s) and a peak-sharpness quality score."""
    n = 1 << int(math.ceil(math.log2(len(ref) + len(sig))))
    X = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X *= ((f > band[0]) & (f < band[1])) / (np.abs(X) + 1e-12)
    cc = np.fft.irfft(X, n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(cc))
    pk = cc[k]
    kf = float(k)
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        kf = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return (kf - max_lag) / fs, pk / (np.mean(np.abs(cc)) + 1e-12)


class AudioScene:
    """Binaural recordings (one per DSLR) + near-range references, full length, band-passed."""

    def __init__(self, src, cams):
        import soundfile as sf
        from scipy.signal import butter, sosfiltfilt
        self.bins, self.refs = [], {}
        for i, s in enumerate(SENSORS):
            x, fs = sf.read(os.path.join(src, f"binaural-{i + 1}.flac"), always_2d=True)
            self.bins.append(x)
        for f in sorted(os.listdir(src)):
            if f.startswith("nearrange"):
                x, fs2 = sf.read(os.path.join(src, f), always_2d=True)
                self.refs[f.replace(".flac", "")] = x.mean(1)
        self.fs = fs
        sos = butter(4, [150, 6000], "bandpass", fs=fs, output="sos")
        self.bins = [sosfiltfilt(sos, b, axis=0) for b in self.bins]
        self.refs = {k: sosfiltfilt(sos, v) for k, v in self.refs.items()}
        self.Cm = np.array([cams[s]["Cn"] for s in SENSORS])
        self.Rm = np.array([cams[s]["Rn"][:, 0] for s in SENSORS])     # camera right axis = ear axis

    def observations(self, t_from, t_to, win=0.25, hop=0.1, dom=4.0):
        """Voiced windows (audio-local time): (actor, t, lavalier->ear delay d[8], quality q[8],
        interaural delay itd[8] (right minus left ear), its quality qi[8])."""
        fs = self.fs
        W, Hh, L, Li = int(win * fs), int(hop * fs), int(0.045 * fs), int(0.0012 * fs)
        names = list(self.refs)
        n = min(min(len(v) for v in self.refs.values()), min(len(b) for b in self.bins)) - W - L
        idx = [i for i in range(int(t_from * fs), min(int(t_to * fs), n), Hh)]
        E = {a: np.array([np.mean(self.refs[a][i:i + W] ** 2) for i in idx]) for a in names}
        floor = {a: np.percentile(E[a], 40) for a in names}
        out = []
        for j, i in enumerate(idx):
            for a in names:
                others = sum(E[b][j] for b in names if b != a) + 1e-12
                if E[a][j] < floor[a] or E[a][j] / others < dom:
                    continue
                ref = self.refs[a][i:i + W]
                d, q, itd, qi = [], [], [], []
                for b in self.bins:
                    r = [gcc_phat(ref, b[i:i + W + L, e], L, fs) for e in range(b.shape[1])]
                    d.append(np.mean([x[0] for x in r]))
                    q.append(min(x[1] for x in r))
                    # interaural delay of the same recorder: right ear relative to left ear
                    t_lr, q_lr = gcc_phat(b[i:i + W, 0], b[i:i + W, 1], 8 * Li, fs, band=(200, 1500))
                    itd.append(t_lr)
                    qi.append(q_lr if abs(t_lr) <= Li / fs else 0.0)   # outside the physical range: reject
                out.append((a, (i + W / 2) / fs, np.array(d), np.array(q), np.array(itd), np.array(qi)))
        return out


def _resid_one(X, m, dl, D, good, Cm):
    dist = np.linalg.norm(X[None] - Cm, axis=1)
    return ((m * dist / C_SOUND + dl - D) * 1e3) * good


def cos_to_ear_axis(X, Cm, Rm):
    v = X[None] - Cm
    return np.sum(v * Rm, 1) / (np.linalg.norm(v, axis=1) + 1e-12)


def solve_window(D, good, Cm, m, dl, X0=None, itd=None, gi=None, Rm=None, b=None, w_itd=3.0):
    """Position for one window: lavalier ranges (fixed scale/latencies) + optional interaural delays
    (itd_m ~ b_m * cos(angle to ear axis)). Returns X, rms range residual (ms)."""
    from scipy.optimize import least_squares
    X0 = np.array([0.0, 0.0, -0.3]) if X0 is None else X0
    use_itd = itd is not None and gi is not None and gi.any()

    def f(X):
        r = _resid_one(X, m, dl, D, good, Cm)
        if use_itd:
            r = np.concatenate([r, w_itd * (b * cos_to_ear_axis(X, Cm, Rm) - itd) * 1e3 * gi])
        return r
    sol = least_squares(f, X0, bounds=(-3, 3), loss="soft_l1", f_scale=0.3)
    r = _resid_one(sol.x, m, dl, D, good, Cm)
    return sol.x, float(np.sqrt(np.sum(r ** 2) / max(good.sum(), 1)))


def fit_positions(obs, Cm, mpu=None, delta=None, q_min=4.0, n_global=200, seed=0):
    """Robust fit of tau_m = mpu*|X_w - C_m|/c + delta.
    Global (mpu, delta) from a joint sparse fit on a subset; then each window solved independently.
    Returns mpu, delta, X (n,3), rms_ms (n,), good (n,8)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    D = np.array([o[2] for o in obs])
    good = np.array([o[3] for o in obs]) > q_min
    M = Cm.shape[0]
    if mpu is None:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(obs), size=min(n_global, len(obs)), replace=False)
        Ds, Gs = D[sub], good[sub]
        nW = len(sub)

        def resid(p):
            m, dl, X = p[0], p[1], p[2:].reshape(nW, 3)
            dist = np.linalg.norm(X[:, None, :] - Cm[None], axis=2)
            return (((m * dist / C_SOUND + dl - Ds) * 1e3) * Gs).ravel()
        J = lil_matrix((nW * M, 2 + 3 * nW), dtype=int)
        for w in range(nW):
            J[w * M:(w + 1) * M, :2] = 1
            J[w * M:(w + 1) * M, 2 + 3 * w: 5 + 3 * w] = 1
        best = None
        for m0 in (2.0, 3.0, 4.5):
            p0 = np.concatenate([[m0, 0.0], np.tile([0.0, 0.0, -0.3], nW)])
            lb = np.concatenate([[0.5, -0.06], np.full(3 * nW, -3.0)])
            ub = np.concatenate([[10.0, 0.06], np.full(3 * nW, 3.0)])
            sol = least_squares(resid, p0, jac_sparsity=J, bounds=(lb, ub), loss="soft_l1", f_scale=0.3,
                                max_nfev=200, x_scale="jac")
            if best is None or sol.cost < best.cost:
                best = sol
        mpu, delta = float(best.x[0]), float(best.x[1])
    X = np.zeros((len(obs), 3))
    rms = np.zeros(len(obs))
    for w in range(len(obs)):
        X[w], rms[w] = solve_window(D[w], good[w], Cm, mpu, delta)
    return mpu, delta, X, rms, good

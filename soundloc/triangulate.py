"""
triangulate.py -- directions from several cameras -> 3D sound-source position

  rays (full direction): point closest to all rays, closed form p = (sum w_i P_i)^-1 sum w_i P_i c_i, P_i = I - d_i d_i^T
  cones (two mics measure only an angle): nonlinear least squares on angle residuals (Gauss-Newton, grid initialisation)
  both kinds can be mixed; Huber weights resist a wrong direction from a single camera (e.g. fooled by a reflection)
  over time: constant-velocity Kalman filter + RTS smoother joins per-frame points into a smooth track
"""
import numpy as np


def closest_point_to_rays(C, D, w=None):
    """C: (n,3) camera centres, D: (n,3) unit directions. Returns p and the condition number of the 3x3 matrix (larger = rays more parallel)"""
    w = np.ones(len(C)) if w is None else np.asarray(w, float)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d, wi in zip(C, D, w):
        P = np.eye(3) - np.outer(d, d)
        A += wi * P
        b += wi * P @ c
    cond = np.linalg.cond(A)
    return np.linalg.solve(A + 1e-9 * np.eye(3), b), cond


def _residuals(p, obs):
    """angle residual of each observation (radians)"""
    r = []
    for o in obs:
        v = p - o["c"]
        n = np.linalg.norm(v) + 1e-9
        if o["kind"] == "dir":
            r.append(np.arccos(np.clip(v @ o["d"] / n, -1, 1)))
        else:
            r.append(np.arccos(np.clip(v @ o["axis"] / n, -1, 1)) - np.arccos(o["cos"]))
    return np.array(r)


def solve(obs, bounds, p0=None, huber=np.deg2rad(5), iters=15):
    """obs: [{"kind":"dir","c","d","w"} or {"kind":"cone","c","axis","cos","w"}]; bounds: (lo, hi) room extent"""
    lo, hi = (np.asarray(b, float) for b in bounds)
    w0 = np.array([o.get("w", 1.0) for o in obs])
    rays = [o for o in obs if o["kind"] == "dir"]
    if p0 is None:
        if len(rays) >= 2:
            p0, _ = closest_point_to_rays(np.array([o["c"] for o in rays]), np.array([o["d"] for o in rays]),
                                          [o.get("w", 1.0) for o in rays])
        else:   # cones only: coarse grid search in the room
            g = np.stack(np.meshgrid(*[np.linspace(lo[k], hi[k], 14) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
            cost = np.array([np.sum(w0 * _residuals(q, obs) ** 2) for q in g])
            p0 = g[int(np.argmin(cost))]
    p = np.clip(np.asarray(p0, float), lo, hi)
    for _ in range(iters):
        r = _residuals(p, obs)
        a = np.abs(r)
        wh = w0 * np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        J = np.zeros((len(obs), 3))
        eps = 1e-4
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = eps
            J[:, k] = (_residuals(p + dp, obs) - r) / eps
        H = J.T @ (wh[:, None] * J) + 1e-6 * np.eye(3)
        step = -np.linalg.solve(H, J.T @ (wh * r))
        p = np.clip(p + step, lo, hi)
        if np.linalg.norm(step) < 1e-5:
            break
    r = _residuals(p, obs)
    return p, float(np.sqrt(np.mean(r ** 2)))


def kalman_smooth(t, Z, valid, q=2.0, r=0.15):
    """Constant-velocity model. t: (T,), Z: (T,3) observations, valid: (T,) bool. q: acceleration noise (m/s^2), r: observation noise (m)"""
    T = len(t)
    xs, Ps, xp, Pp = np.zeros((T, 6)), np.zeros((T, 6, 6)), np.zeros((T, 6)), np.zeros((T, 6, 6))
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    R = r ** 2 * np.eye(3)
    first = int(np.argmax(valid)) if valid.any() else 0
    x = np.concatenate([Z[first] if valid.any() else np.zeros(3), np.zeros(3)])
    P = np.diag([1, 1, 1, 4, 4, 4.0])
    for k in range(T):
        dt = t[k] - t[k - 1] if k else 0.0
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        G = np.vstack([0.5 * dt ** 2 * np.eye(3), dt * np.eye(3)])
        x = F @ x
        P = F @ P @ F.T + q ** 2 * G @ G.T
        xp[k], Pp[k] = x, P
        if valid[k]:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (Z[k] - H @ x)
            P = (np.eye(6) - K @ H) @ P
        xs[k], Ps[k] = x, P
    # RTS smoother
    for k in range(T - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        A = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + A @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + A @ (Ps[k + 1] - Pp[k + 1]) @ A.T
    return xs[:, :3]

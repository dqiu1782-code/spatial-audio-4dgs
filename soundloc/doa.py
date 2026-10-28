"""
doa.py -- per camera, estimate the direction the sound comes from (DOA)

  GCC-PHAT: time difference of arrival tau between two mics (fairly robust to reverberation)
  SRP-PHAT: with >= 3 mics, sum GCC-PHAT over all mic pairs on a grid of directions and take the maximum
            -> full 3D unit direction in camera coordinates
  with only 2 mics: only the angle to the mic axis is known, i.e. a cone (front/back and up/down are ambiguous)

Convention: u is the unit vector from the array to the source; arrival time at mic k is t_k = -m_k.u / c (far-field approximation)
"""
import numpy as np

from geom import C_SOUND


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


class GCC:
    """GCC-PHAT for all mic pairs of one multichannel frame (zero-padded in frequency for sub-sample cross-correlation)"""

    def __init__(self, nfft, fs, up=8, fmin=200.0, fmax=8000.0):
        self.nfft, self.fs, self.up = nfft, fs, up
        f = np.fft.rfftfreq(2 * nfft, 1 / fs)
        self.band = ((f >= fmin) & (f <= fmax)).astype(float)
        self.win = np.hanning(nfft)

    def pairs(self, frame, pairs):
        """frame: (nfft, M); returns {(i,j): (cc, centre index)}, cc[k] corresponds to delay (k - centre)/(fs*up) s"""
        X = np.fft.rfft(frame * self.win[:, None], 2 * self.nfft, axis=0)
        out = {}
        L = 2 * self.nfft * self.up
        for i, j in pairs:
            G = X[:, i] * np.conj(X[:, j])
            G = G / (np.abs(G) + 1e-12) * self.band
            cc = np.fft.irfft(G, L)
            cc = np.concatenate([cc[-L // 2:], cc[:L // 2]])
            out[(i, j)] = cc
        return out, L // 2


def all_pairs(M):
    return [(i, j) for i in range(M) for j in range(i + 1, M)]


class DOAEstimator:
    def __init__(self, mics_local, fs, nfft=4096, grid=2562, up=8, fmin=200.0, fmax=8000.0):
        self.m = np.asarray(mics_local, float)
        self.M = len(self.m)
        self.fs, self.nfft, self.up = fs, nfft, up
        self.gcc = GCC(nfft, fs, up, fmin, fmax)
        self.pairs = all_pairs(self.M)
        self.U = fib_sphere(grid)
        # theoretical delay of each mic pair for each direction (index offset after upsampling)
        self.lag = {}
        for i, j in self.pairs:
            tau = -(self.m[i] - self.m[j]) @ self.U.T / C_SOUND
            self.lag[(i, j)] = tau * fs * up
        self.rank = np.linalg.matrix_rank(self.m - self.m.mean(0), tol=1e-4)

    def _srp(self, ccs, c0, U, lags):
        P = np.zeros(len(U))
        for pr, cc in ccs.items():
            x = lags[pr] + c0
            i0 = np.clip(np.floor(x).astype(int), 0, len(cc) - 2)
            fr = x - i0
            P += cc[i0] * (1 - fr) + cc[i0 + 1] * fr
        return P / len(ccs)

    def estimate(self, frame):
        """Returns a dict:
           kind='dir'  -> u (unit direction in camera frame), score
           kind='cone' -> axis (unit vector in camera frame), cos (value of u.axis), score"""
        ccs, c0 = self.gcc.pairs(frame, self.pairs)
        if self.M == 2 or self.rank < 2:
            (i, j), cc = next(iter(ccs.items()))
            d = np.linalg.norm(self.m[i] - self.m[j])
            maxlag = int(np.ceil(d / C_SOUND * self.fs * self.up)) + 2
            seg = cc[c0 - maxlag:c0 + maxlag + 1]
            k = int(np.argmax(seg))
            if 0 < k < len(seg) - 1:                       # parabolic interpolation
                y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
                k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
            tau = (k - maxlag) / (self.fs * self.up)
            axis = (self.m[i] - self.m[j]) / d
            cosv = float(np.clip(-C_SOUND * tau / d, -1, 1))
            return {"kind": "cone", "axis": axis, "cos": cosv, "score": float(np.max(seg))}
        P = self._srp(ccs, c0, self.U, self.lag)
        k = int(np.argmax(P))
        u = self.U[k]
        # refine: a denser ring of directions around the maximum
        for rad in (0.08, 0.025):
            cand = u + rad * np.random.default_rng(k).standard_normal((300, 3))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand = np.vstack([u, cand])
            lags = {pr: -(self.m[pr[0]] - self.m[pr[1]]) @ cand.T / C_SOUND * self.fs * self.up for pr in self.pairs}
            Pc = self._srp(ccs, c0, cand, lags)
            u = cand[int(np.argmax(Pc))]
        score = float(np.max(P))
        if self.rank == 2:      # planar array: cannot tell which side of the plane -- take the one in front of the camera (+z)
            n = np.linalg.svd(self.m - self.m.mean(0))[2][-1]
            if u @ n * n[2] < 0 and abs(u @ n) > 1e-3:
                u = u - 2 * (u @ n) * n
        return {"kind": "dir", "u": u, "score": score}


def frames(x, nfft, hop):
    """x: (n, M) -> list of frame start indices"""
    return list(range(0, x.shape[0] - nfft, hop))

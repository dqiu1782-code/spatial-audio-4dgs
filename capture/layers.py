"""Voice / ambience layering of real recordings.

For every mic channel y_k the voices are modelled with a convolutive transfer function (CTF) in
the STFT domain:  Y_k(f, n) = sum_a sum_l H_kal(f) X_a(f, n - l) + A_k(f, n)
where X_a is a clean reference of talker a (lavalier, or a beamformer steered at the talker's
video position). H is re-estimated on overlapping blocks (talkers move). The residual A_k is the
ambience layer: room tone, AC, doors, footsteps, crew; it keeps the recording's real spatial cues.

Playback (viewer): voices = reference signals rendered with HRTF at the tracked positions;
ambience = residual of the mic(s) nearest to the listener, mixed in as a bed.
"""
import numpy as np
from scipy.signal import istft, stft

C = 343.0


def ctf_separate(Y, refs, fs, nfft=1024, hop=256, taps=12, block_s=3.0, lam=1e-3):
    """Y: (n, K) mic channels; refs: (n, A) talker references. Returns (voice_est, ambience), both (n, K)."""
    n, K = Y.shape
    _, _, Zy = stft(Y.T, fs, nperseg=nfft, noverlap=nfft - hop)          # (K, F, T)
    _, _, Zx = stft(refs.T, fs, nperseg=nfft, noverlap=nfft - hop)       # (A, F, T)
    A, F, T = Zx.shape
    # regressors: every talker x every tap -> (F, T, A*taps)
    R = np.zeros((F, T, A * taps), complex)
    for a in range(A):
        for l in range(taps):
            R[:, l:, a * taps + l] = Zx[a, :, :T - l]
    blk = max(int(block_s * fs / hop), 4 * A * taps)
    step = blk // 2
    win = np.hanning(blk + 2)[1:-1]
    V = np.zeros((K, F, T), complex)
    wsum = np.zeros(T)
    for s in range(0, max(T - blk, 0) + step, step):
        e = min(s + blk, T)
        w = win[: e - s]
        Rb = R[:, s:e]                                                  # (F, t, P)
        G = np.einsum("ftp,ftq->fpq", Rb.conj(), Rb)                    # normal equations
        reg = lam * np.trace(G, axis1=1, axis2=2).real[:, None, None] / G.shape[1] + 1e-12
        G = G + reg * np.eye(G.shape[1])[None]
        for k in range(K):
            rhs = np.einsum("ftp,ft->fp", Rb.conj(), Zy[k, :, s:e])
            H = np.linalg.solve(G, rhs[..., None])[..., 0]              # (F, P)
            V[k, :, s:e] += np.einsum("ftp,fp->ft", Rb, H) * w[None]
        wsum[s:e] += w
    V /= np.maximum(wsum, 1e-6)[None, None]
    _, v = istft(V, fs, nperseg=nfft, noverlap=nfft - hop)
    v = v.T[:n]
    if len(v) < n:
        v = np.pad(v, ((0, n - len(v)), (0, 0)))
    return v, Y - v


def beamform_refs(rig, mics, offsets, params, tracks, fs, ch_positions):
    """Delay-and-sum voice references steered at each talker's (video) position, for rigs without
    lavaliers. mics: {mic_id: (n, ch)}; offsets: clap-sync offsets; params: calibrated per-device."""
    sigs, pos, lat, drift = [], [], [], []
    for i, m in enumerate(rig.mics):
        for j in range(mics[m.id].shape[1]):
            sigs.append((mics[m.id][:, j], offsets[m.id]))
            lat.append(params[i]["lat"])
            drift.append(params[i]["drift"])
    n = min(len(s) for s, _ in sigs)
    t = np.arange(n) / fs
    out = np.zeros((n, len(tracks)))
    for a, tr in enumerate(tracks):
        X = np.stack([np.interp(t, tr["t"], np.asarray(tr["pos"])[:, j]) for j in range(3)], 1)
        acc = np.zeros(n)
        dmin = np.min(np.linalg.norm(X[:, None] - ch_positions[None], axis=2), axis=1)
        for k, (s, off) in enumerate(sigs):
            d = np.linalg.norm(X - ch_positions[k], axis=1)
            idx = (t + off + (d - dmin) / C + lat[k] + drift[k] * t) * fs   # arrival at mic k
            acc += np.interp(idx, np.arange(len(s)), s, left=0, right=0) * (d / dmin)
        out[:, a] = acc / len(sigs)
    return out


def activity_gate(t_win, actor, n_talkers, n, fs, win=0.25, dilate_s=0.4, ramp_s=0.1):
    """Soft per-talker activity (n, A) from self-cal attribution (actor[w] = talker index or -1).
    Used to gate beamformed references so background between turns is not explained as voice."""
    g = np.zeros((n, n_talkers))
    for tw, a in zip(t_win, actor):
        if a >= 0:
            i0, i1 = int((tw - win / 2 - dilate_s) * fs), int((tw + win / 2 + dilate_s) * fs)
            g[max(i0, 0):max(min(i1, n), 0), a] = 1.0
    k = max(int(ramp_s * fs), 1)
    ker = np.hanning(2 * k + 1)
    ker /= ker.sum()
    return np.stack([np.convolve(g[:, a], ker, "same") for a in range(n_talkers)], 1)


def leakage_db(est_ambience, true_background, true_voice):
    """Energy of what is wrong in the ambience layer relative to the voice energy (lower is better)."""
    err = est_ambience - true_background
    return 10 * np.log10(np.sum(err ** 2) / np.sum(true_voice ** 2))

"""Clap-slate sync across independent recorders.

Every recording (each recorder's first channel, each lavalier) is reduced to a high-frequency
onset envelope; the declared clap pattern (e.g. 3 claps) is located in the first seconds, then
refined by GCC-PHAT on the raw audio around the first clap. Known propagation from the clap
position to each mic is removed, giving each recorder's offset: t_recorded = t_physical + offset.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

C = 343.0


def onset_envelope(x, fs, hop_s=0.002):
    sos = butter(4, min(2000, 0.4 * fs), "highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x) ** 2
    h = int(hop_s * fs)
    e = np.log10(np.add.reduceat(y[: len(y) // h * h], np.arange(0, len(y) // h * h, h)) + 1e-12)
    return np.maximum(np.diff(e, prepend=e[0]), 0), h


def find_claps(x, fs, n_claps=3, search_s=30.0, gap_s=(0.25, 1.5)):
    """Times (s) of the clap pattern: n strong onsets with regular spacing (gap ratio <= 1.6).
    Among all such runs in the search window the one whose weakest clap is strongest wins, so
    louder single events (doors, speech) elsewhere in the window do not break the pattern."""
    env, h = onset_envelope(x[: int(search_s * fs)], fs)
    dt = h / fs
    pk, prop = find_peaks(env, distance=max(int(0.15 / dt), 1), height=np.percentile(env, 99))
    if len(pk) < n_claps:
        raise ValueError("clap pattern not found")
    hgt = prop["peak_heights"]
    keep = hgt >= 0.25 * hgt.max()
    pk, hgt = pk[keep], hgt[keep]
    best, best_score = None, -1.0

    def extend(run):
        nonlocal best, best_score
        if len(run) == n_claps:
            score = min(hgt[i] for i in run)
            if score > best_score * 1.05:                 # prefer the earliest of near-equal runs
                best, best_score = list(run), score
            return
        for j in range(run[-1] + 1, len(pk)):
            g = (pk[j] - pk[run[-1]]) * dt
            if g > gap_s[1]:
                break
            if g < gap_s[0]:
                continue
            if len(run) >= 2:
                g0 = (pk[run[-1]] - pk[run[-2]]) * dt
                if max(g, g0) / min(g, g0) > 1.6:
                    continue
            extend(run + [j])

    for i in range(len(pk)):
        extend([i])
    if best is None:
        raise ValueError("clap pattern not found")
    return pk[best] * dt


def refine(ref, x, fs, t_ref, t_x, half=0.05):
    """Sub-sample time of the clap in x given its time in ref (GCC-PHAT around both)."""
    a, b = int((t_ref - half) * fs), int((t_x - 2 * half) * fs)
    r, s = ref[a: a + int(2 * half * fs)], x[max(b, 0): b + int(4 * half * fs)]
    n = 1 << int(np.ceil(np.log2(len(r) + len(s))))
    X = np.fft.rfft(s, n) * np.conj(np.fft.rfft(r, n))
    cc = np.fft.irfft(X / (np.abs(X) + 1e-12), n)
    L = len(s)
    cc = np.concatenate([cc[-len(r):], cc[:L]])
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    lag = (k - len(r)) / fs                        # s(t) ~ r(t - lag), segment-local
    return max(b, 0) / fs + (t_ref - a / fs) + lag  # absolute clap time in x


def sync_recordings(recs, fs, clap_pos, mic_pos, n_claps=3, ref=None):
    """recs: {name: mono signal}; mic_pos: {name: nominal position (3,)} (lavaliers: talker position).
    Returns {name: offset_s} with t_recorded = t_physical + offset, physical time = reference's clock."""
    times = {k: find_claps(v, fs, n_claps) for k, v in recs.items()}
    ref = ref or next(iter(recs))
    t_ref = times[ref][0]
    out = {}
    for k, v in recs.items():
        t = t_ref if k == ref else refine(recs[ref], v, fs, t_ref, times[k][0])
        out[k] = t - np.linalg.norm(np.asarray(mic_pos[k]) - clap_pos) / C
    base = out[ref]
    return {k: v - base for k, v in out.items()}, times

"""Coarse offsets between all audio files of a scene (relative to binaural-1), via long-window GCC-PHAT."""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_common import gcc_phat

src = sys.argv[1]
fs_lo = 8000
files = [f"binaural-{i}.flac" for i in range(1, 9)] + sorted(f for f in os.listdir(src) if f.startswith("nearrange"))
sig = {}
for f in files:
    x, fs = sf.read(os.path.join(src, f), always_2d=True)
    sig[f] = resample_poly(x.mean(1), fs_lo, fs)
    print(f"{f:28s} {len(x) / fs:7.2f} s", flush=True)
ref = sig["binaural-1.flac"]
max_lag = int(8 * fs_lo)
for f in files[1:]:
    res = []
    for start in (30, 70, 110):
        a = int(start * fs_lo)
        r = ref[a:a + 20 * fs_lo]
        s = sig[f][max(0, a - max_lag): a + 20 * fs_lo + max_lag]
        pad = max_lag - (a - max(0, a - max_lag))
        s = np.concatenate([np.zeros(pad), s])
        d, q = gcc_phat(r, s, 2 * max_lag, fs_lo, band=(200, 3500))
        res.append((d - max_lag / fs_lo, q))
    print(f"{f:28s} offset vs binaural-1: " + "  ".join(f"{d * 1e3:9.1f} ms (q {q:5.1f})" for d, q in res), flush=True)
print("OFFSETS_DONE")

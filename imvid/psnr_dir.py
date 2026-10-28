"""Mean PSNR of test renders vs GT for a 4DGaussians output dir (metrics.py needs pytorch_msssim)."""
import glob
import sys

import numpy as np
from PIL import Image

d = sorted(glob.glob(sys.argv[1] + r"\test\ours_*"))[-1]
L = lambda f: np.asarray(Image.open(f).convert("RGB"), np.float32) / 255      # noqa: E731
p = [-10 * np.log10(((L(a) - L(b)) ** 2).mean()) for a, b in zip(sorted(glob.glob(d + r"\renders\*.png")),
                                                               sorted(glob.glob(d + r"\gt\*.png")))]
print(f"{d}: {len(p)} frames, PSNR mean {np.mean(p):.2f} min {np.min(p):.2f} max {np.max(p):.2f}")

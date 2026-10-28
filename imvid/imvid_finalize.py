"""Move the prepared ImViD dataset into a metric world: origin on the floor under the singer, z up,
metres (scale from imvid_track.py). Rewrites transforms_*.json, braindance_meta.json and
face_track.json in place (idempotent: skipped if already metric).

  python imvid_finalize.py DATA
"""
import json
import os
import sys

import numpy as np


def main():
    data = sys.argv[1]
    mp = os.path.join(data, "braindance_meta.json")
    meta = json.load(open(mp))
    if meta.get("metric"):
        print("already metric")
        return
    tr = json.load(open(os.path.join(data, "face_track.json")))
    s = tr["metres_per_unit"]
    feet = np.array(tr["feet"])
    o = np.r_[np.median(feet[:, :2], 0), np.median(feet[:, 2])]
    f = lambda X: ((np.asarray(X) - o) * s).tolist()           # noqa: E731
    for split in ("train", "test"):
        p = os.path.join(data, f"transforms_{split}.json")
        T = json.load(open(p))
        for fr in T["frames"]:
            M = np.array(fr["transform_matrix"])
            M[:3, 3] = f(M[:3, 3])
            fr["transform_matrix"] = M.tolist()
        json.dump(T, open(p, "w"), indent=1)
    for c in meta["cameras"].values():
        c["C"] = f(c["C"])
    meta["metric"] = {"origin_prep_units": o.tolist(), "metres_per_unit": s, "note": tr["assumption"]}
    meta["coords"] = "metres, z up, origin on the floor under the singer, +x towards the cameras"
    json.dump(meta, open(mp, "w"), indent=1)
    for k in ("head_top", "feet", "mouth"):
        tr[k] = f(tr[k])
    tr["units"] = "metres (final world)"
    json.dump(tr, open(os.path.join(data, "face_track.json"), "w"), indent=1)
    C = np.array([c["C"] for c in meta["cameras"].values()])
    print(f"origin {o.round(3)} (prep units), scale {s:.4f} m/unit")
    print(f"cameras: x {C[:, 0].min():.2f}..{C[:, 0].max():.2f}  y {C[:, 1].min():.2f}..{C[:, 1].max():.2f}  "
          f"z {C[:, 2].min():.2f}..{C[:, 2].max():.2f} m; mouth median {np.median(tr['mouth'], 0).round(2)} m")


if __name__ == "__main__":
    main()

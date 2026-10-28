"""Metric scale for a COLMAP reconstruction from known real distances.

COLMAP poses are correct up to one global scale. One tape-measured distance between two cameras
(or between two points visible in the images) fixes it for the whole scene; several distances
give a least-squares scale plus a consistency check.

  python scale.py MODEL --pair cam00 cam05 0.42 [--pair camA camB metres ...]
                        [--rig rig.json]                  use every camera pair from rig positions (metres)
                        [--measure "cam00:812,430 cam07:790,433" "cam00:900,610 cam07:880,615"]
                        [--out SCALED_MODEL]              write the model in metres

MODEL is a COLMAP text model dir (cameras.txt, images.txt, optional points3D.txt).
--measure takes two points, each as "image:u,v image:u,v ..." (>= 2 images, pixel coords in the
original images); prints their distance in metres.
"""
import argparse
import json
import os

import numpy as np


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_model(d):
    cams = {}
    for l in open(os.path.join(d, "cameras.txt")):
        if l.startswith("#") or not l.strip():
            continue
        s = l.split()
        cams[int(s[0])] = (s[1], int(s[2]), int(s[3]), np.array(list(map(float, s[4:]))))
    imgs, lines = {}, [l for l in open(os.path.join(d, "images.txt")) if not l.startswith("#")]
    for l in lines:
        s = l.split()
        if len(s) == 10:
            name = os.path.splitext(s[9])[0]
            imgs[name] = {"q": list(map(float, s[1:5])), "t": np.array(list(map(float, s[5:8]))), "cam": int(s[8]),
                          "file": s[9], "line": l}
            imgs[name]["R"] = q2R(imgs[name]["q"])
            imgs[name]["C"] = -imgs[name]["R"].T @ imgs[name]["t"]
    return cams, imgs


def normalized_ray(cam, img, u, v):
    """World-space unit ray through pixel (u, v), lens distortion removed."""
    model, W, H, p = cam
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        fx = fy = f
        dist = np.r_[p[3:5], 0, 0][:4] if model != "SIMPLE_PINHOLE" else np.zeros(4)
    else:                                                   # PINHOLE, OPENCV
        fx, fy, cx, cy = p[:4]
        dist = np.r_[p[4:8], np.zeros(4)][:4]
    x, y = (u - cx) / fx, (v - cy) / fy
    if np.any(dist):
        import cv2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        x, y = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist)[0, 0]
    d = img["R"].T @ np.array([x, y, 1.0])
    return d / np.linalg.norm(d)


def triangulate(rays):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for C, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b)


def fit_scale(imgs, pairs):
    """pairs: [(a, b, metres)] -> (metres per unit, per-pair table)."""
    du = np.array([np.linalg.norm(imgs[a]["C"] - imgs[b]["C"]) for a, b, _ in pairs])
    dm = np.array([m for _, _, m in pairs])
    s = float(du @ dm / (du @ du))
    rows = [(a, b, m, u, m / u, (s * u - m) * 100) for (a, b, m), u in zip(pairs, du)]
    return s, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pair", nargs=3, action="append", default=[], metavar=("IMG_A", "IMG_B", "METRES"))
    ap.add_argument("--rig", default="", help="rig.json with camera positions in metres (names must match)")
    ap.add_argument("--measure", nargs=2, default=None, metavar=("POINT1", "POINT2"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cams, imgs = read_model(a.model)
    pairs = [(x, y, float(m)) for x, y, m in a.pair]
    if a.rig:
        rig = json.load(open(a.rig, encoding="utf-8"))
        pos = {c["name"]: np.array(c["pos"], float) for c in rig["cameras"] if c["name"] in imgs}
        ns = sorted(pos)
        pairs += [(ns[i], ns[j], float(np.linalg.norm(pos[ns[i]] - pos[ns[j]])))
                  for i in range(len(ns)) for j in range(i + 1, len(ns))]
    if not pairs:
        raise SystemExit("need at least one --pair or --rig")
    s, rows = fit_scale(imgs, pairs)
    print(f"scale: {s:.5f} m per COLMAP unit ({len(pairs)} distance(s))")
    if len(rows) > 1:
        err = np.array([r[5] for r in rows])
        print(f"  consistency: residual median {np.median(np.abs(err)):.1f} cm, max {np.max(np.abs(err)):.1f} cm")
    for a_, b_, m, u, si, e in rows[:12]:
        print(f"  {a_}-{b_}: {m:.3f} m = {u:.4f} units -> {si:.5f} m/unit, residual {e:+.1f} cm")
    if a.measure:
        P = []
        for spec in a.measure:
            rays = []
            for tok in spec.split():
                name, uv = tok.split(":")
                u, v = map(float, uv.split(","))
                img = imgs[os.path.splitext(name)[0]]
                rays.append((img["C"], normalized_ray(cams[img["cam"]], img, u, v)))
            P.append(triangulate(rays))
        print(f"measured distance: {np.linalg.norm(P[0] - P[1]) * s:.3f} m")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.model, "cameras.txt")) as f, open(os.path.join(a.out, "cameras.txt"), "w") as g:
            g.write(f.read())
        with open(os.path.join(a.out, "images.txt"), "w") as g:
            g.write(f"# scaled to metres ({s:.6f} m/unit)\n")
            for name, im in imgs.items():
                t = im["t"] * s
                g.write(f"{im['line'].split()[0]} {' '.join(map(str, im['q']))} "
                        f"{' '.join(map(str, t))} {im['cam']} {im['file']}\n\n")
        p3 = os.path.join(a.model, "points3D.txt")
        if os.path.isfile(p3):
            with open(p3) as f, open(os.path.join(a.out, "points3D.txt"), "w") as g:
                for l in f:
                    s_ = l.split()
                    if l.startswith("#") or len(s_) < 4:
                        g.write(l)
                        continue
                    xyz = np.array(list(map(float, s_[1:4]))) * s
                    g.write(" ".join([s_[0]] + [repr(v) for v in xyz] + s_[4:]) + "\n")
        print(f"scaled model -> {a.out}")


if __name__ == "__main__":
    main()

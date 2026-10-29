"""Singer track + metric scale for the ImViD Opera sample.

- Person box per view: YOLOX-s (Megvii, Apache-2.0, ONNX via cv2.dnn).
- Head top (box top centre) and feet (box bottom centre) triangulated per frame with RANSAC.
- Scale: standing height assumed 1.62 m. Mouth = head top - 0.14 m along the floor normal.

  python imvid_track.py DATA MODEL.onnx
Writes DATA/face_track.json (prep units + metres_per_unit) and DATA/track_overlay.jpg.
"""
import json
import os
import sys

import cv2
import numpy as np

HEIGHT_M, MOUTH_DOWN = 1.62, 0.14


class Yolox:
    def __init__(self, path, size=640):
        self.net, self.size = cv2.dnn.readNetFromONNX(path), size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            yv, xv = np.mgrid[0:n, 0:n]
            grids.append(np.stack([xv, yv], 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def person(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))
        blob = pad.transpose(2, 0, 1)[None].astype(np.float32)
        self.net.setInput(blob)
        o = self.net.forward()[0]
        xy = (o[:, :2] + self.grid) * self.stride
        wh = np.exp(o[:, 2:4]) * self.stride
        sc = o[:, 4] * o[:, 5]                              # objectness * person
        i = np.argmax(sc)
        if sc[i] < 0.4:
            return None
        cx, cy = xy[i] / r
        bw, bh = wh[i] / r
        return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), float(sc[i])


def tri(obs):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for C, d in obs:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b)


def proj(X, R, C, K):
    p = K @ (R.T @ (X - C))
    return p[:2] / p[2], p[2]


def ransac_point(cand, cam, K, tol, rng):
    best = []
    for _ in range(300):
        if len(cand) < 2:
            break
        i, j = rng.choice(len(cand), 2, replace=False)
        X = tri([cand[i][1:3], cand[j][1:3]])
        inl = []
        for c in cand:
            p, z = proj(X, cam[c[0]][0], c[1], K)
            if z > 0 and np.linalg.norm(p - c[3]) < tol:
                inl.append(c)
        if len(inl) > len(best):
            best = inl
    if len(best) < 4:
        return None, 0, np.nan
    X = tri([c[1:3] for c in best])
    e = np.median([np.linalg.norm(proj(X, cam[c[0]][0], c[1], K)[0] - c[3]) for c in best])
    return X, len(best), float(e)


def main():
    data, model = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(data, "scene_meta.json")))
    names = sorted(meta["cameras"])
    n_t = meta["n_times"]
    dd = os.path.join(data, "_det")
    Wd = cv2.imread(os.path.join(dd, f"{names[0]}_t000.jpg")).shape[1]
    K = np.array(meta["K"]) * (Wd / meta["w"])
    K[2, 2] = 1
    Ki = np.linalg.inv(K)
    cam = {n: (np.array(meta["cameras"][n]["R"]), np.array(meta["cameras"][n]["C"])) for n in names}
    cache = os.path.join(data, "_boxes.json")
    boxes = {}
    if os.path.isfile(cache):
        boxes = {tuple(k.split("|")[:1]) + (int(k.split("|")[1]),): np.array(v) for k, v in json.load(open(cache)).items()}
    det = None if boxes else Yolox(model)
    for n in (names if det else []):
        probe = [det.person(cv2.imread(os.path.join(dd, f"{n}_t{k:03d}.jpg"))) for k in (0, n_t // 2, n_t - 1)]
        if all(r is None for r in probe):                  # outward rig: most cameras never see the singer
            print(n, "no person", flush=True)
            continue
        for k in range(n_t):
            r = det.person(cv2.imread(os.path.join(dd, f"{n}_t{k:03d}.jpg")))
            if r is not None:
                boxes[(n, k)] = r[0]
        print(n, sum((n, k) in boxes for k in range(n_t)), "boxes", flush=True)
    if det:
        json.dump({f"{n}|{k}": b.tolist() for (n, k), b in boxes.items()}, open(cache, "w"))

    def ray(n, u):
        d = cam[n][0] @ (Ki @ np.r_[u, 1])
        return d / np.linalg.norm(d)

    rng = np.random.default_rng(0)
    T, HEAD, FEET, NV, RE = [], [], [], [], []
    for k in range(n_t):
        ch, cf = [], []
        for n in names:
            if (n, k) in boxes:
                x0, y0, x1, y1 = boxes[(n, k)]
                uh, uf = np.array([(x0 + x1) / 2, y0]), np.array([(x0 + x1) / 2, y1])
                mg, Hd = 0.01 * Wd, Wd * meta["h"] / meta["w"]
                if x0 < mg or x1 > Wd - mg:                     # cut at the side: centre unreliable
                    continue
                if y0 > mg:
                    ch.append((n, cam[n][1], ray(n, uh), uh))
                if y1 < Hd - mg:
                    cf.append((n, cam[n][1], ray(n, uf), uf))
        Xh, nh, eh = ransac_point(ch, cam, K, 0.015 * Wd, rng)
        Xf, nf, ef = ransac_point(cf, cam, K, 0.015 * Wd, rng)
        if Xh is None or Xf is None:
            continue
        T.append(k / meta["fps"])
        HEAD.append(Xh)
        FEET.append(Xf)
        NV.append((nh, nf))
        RE.append((eh, ef))
    HEAD, FEET = np.array(HEAD), np.array(FEET)
    up = np.array([0, 0, 1.0])
    Hu = float(np.median((HEAD - FEET) @ up))
    mpu = HEIGHT_M / Hu
    nv = np.array(NV)
    print(f"{len(T)}/{n_t} frames; views head {np.median(nv[:, 0]):.0f} feet {np.median(nv[:, 1]):.0f} of {len(names)}; "
          f"reproj median {np.nanmedian(np.array(RE)):.1f} px at {Wd}")
    print(f"height {Hu:.3f} units -> {mpu:.4f} m/unit; head-feet horizontal offset "
          f"{np.median(np.linalg.norm((HEAD - FEET)[:, :2], axis=1)) * mpu * 100:.0f} cm")
    Cs = np.array([cam[n][1] for n in names])
    floor_z = float(np.median(FEET[:, 2]))
    print(f"singer feet at {np.median(FEET, 0).round(3)}; cameras: centroid {Cs.mean(0).round(3)}, "
          f"spread {np.ptp(Cs, 0).round(3)} units = {(np.ptp(Cs, 0) * mpu).round(2)} m, "
          f"height above floor {(np.median(Cs[:, 2]) - floor_z) * mpu:.2f} m, "
          f"distance to singer {np.median(np.linalg.norm(Cs - np.median(HEAD, 0), axis=1)) * mpu:.2f} m")
    mouth = HEAD - up * (MOUTH_DOWN / mpu)
    json.dump({"t": T, "head_top": HEAD.tolist(), "feet": FEET.tolist(), "mouth": mouth.tolist(),
               "n_views": NV, "reproj_px": RE, "det_width": Wd, "height_units": Hu, "metres_per_unit": mpu,
               "floor_z_units": floor_z, "assumption": f"standing height {HEIGHT_M} m, mouth {MOUTH_DOWN} m below head top"},
              open(os.path.join(data, "face_track.json"), "w"), indent=1)
    k = n_t // 2
    i = int(np.argmin(np.abs(np.array(T) - k / meta["fps"])))
    tiles = []
    for n in names[::7][:6]:
        im = cv2.imread(os.path.join(dd, f"{n}_t{k:03d}.jpg"))
        if (n, k) in boxes:
            x0, y0, x1, y1 = boxes[(n, k)].astype(int)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), 3)
        for X, col in ((mouth[i], (0, 0, 255)), (FEET[i], (255, 0, 0))):
            p, _ = proj(X, cam[n][0], cam[n][1], K)
            cv2.drawMarker(im, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 50, 4)
        cv2.putText(im, n, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 4)
        tiles.append(cv2.resize(im, (640, 360)))
    cv2.imwrite(os.path.join(data, "track_overlay.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]))


if __name__ == "__main__":
    main()

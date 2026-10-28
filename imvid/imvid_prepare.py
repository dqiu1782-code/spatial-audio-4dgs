"""Convert the ImViD Opera sample (39 GoPros, 5 s, 60 fps, stereo audio per camera) into
4DGaussians Blender/D-NeRF format + a sound scene for the Braindance viewer.

World: origin at the cameras' convergence point, z up (mean camera up), COLMAP units scaled
by --mpu (metres per COLMAP unit, from imvid_scale.py) so that the output is in metres.
Images: undistorted (OPENCV model) to one shared pinhole K at --width.

  python imvid_prepare.py SRC OUT [--fps 10 --width 960 --test cam00 --mpu 1.0]
"""
import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_colmap(src):
    cam = [l.split() for l in open(os.path.join(src, "cameras.txt")) if l.strip() and not l.startswith("#")][0]
    W, H = int(cam[2]), int(cam[3])
    fx, fy, cx, cy, k1, k2, p1, p2 = map(float, cam[4:12])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.array([k1, k2, p1, p2])
    cams = {}
    for l in open(os.path.join(src, "images.txt")):
        s = l.split()
        if l.startswith("#") or len(s) != 10:
            continue
        R = q2R(np.array(list(map(float, s[1:5]))))
        t = np.array(list(map(float, s[5:8])))
        cams[os.path.splitext(s[9])[0]] = {"R": R, "t": t}
    return W, H, K, dist, cams


def normalise(cams):
    """Similarity to a z-up world centred on the point the optical axes converge to."""
    P = np.array([-c["R"].T @ c["t"] for c in cams.values()])
    F = np.array([c["R"][2] for c in cams.values()])
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, f in zip(P, F):
        M = np.eye(3) - np.outer(f, f)
        A += M
        b += M @ p
    X = np.linalg.solve(A, b)
    up = -np.mean([c["R"][1] for c in cams.values()], 0)
    up /= np.linalg.norm(up)
    fwd = np.mean(F, 0)                                     # mean viewing direction -> world +x ... flattened
    fwd -= up * fwd.dot(up)
    fwd /= np.linalg.norm(fwd)
    xa = -fwd                                               # +x points from the stage towards the cameras
    ya = np.cross(up, xa)
    Rw = np.stack([xa, ya, up])                             # rows: new axes in colmap coords
    return X, Rw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--det_width", type=int, default=1920, help="also keep undistorted frames this wide for detection (0: off)")
    ap.add_argument("--test", default="cam00")
    ap.add_argument("--mpu", type=float, default=1.0)
    a = ap.parse_args()
    W0, H0, K0, dist, cams = load_colmap(a.src)
    X, Rw = normalise(cams)
    s = a.mpu
    Wt = a.width
    Ht = int(round(H0 * Wt / W0 / 2) * 2)
    f = (K0[0, 0] + K0[1, 1]) / 2 * Wt / W0
    Kt = np.array([[f, 0, Wt / 2], [0, f, Ht / 2], [0, 0, 1.0]])
    fovx = 2 * math.atan(Wt / (2 * f))
    names = sorted(cams)
    n_t = int(round(5.0 * a.fps))
    frames = {"train": [], "test": []}
    world = {}
    for n in names:
        R, t = cams[n]["R"], cams[n]["t"]
        Rn = (Rw @ R.T)                                     # cam->world rotation (cols: right, down, fwd)
        Cn = Rw @ (-R.T @ t - X) * s
        world[n] = (Rn, Cn)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = Rn[:, 0], -Rn[:, 1], -Rn[:, 2], Cn
        split = "test" if n == a.test else "train"
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        jobs = [(Wt, Ht, Kt, os.path.join(a.out, split), f"{n}_t%03d.jpg")]
        if a.det_width:
            Wd, Hd = a.det_width, int(round(H0 * a.det_width / W0 / 2) * 2)
            fd = f * Wd / Wt
            jobs.append((Wd, Hd, np.array([[fd, 0, Wd / 2], [0, fd, Hd / 2], [0, 0, 1.0]]),
                         os.path.join(a.out, "_det"), f"{n}_t%03d.jpg"))
            os.makedirs(jobs[-1][3], exist_ok=True)
        maps = [cv2.initUndistortRectifyMap(K0, dist, None, Kj, (Wj, Hj), cv2.CV_32FC1) + (d, p)
                for Wj, Hj, Kj, d, p in jobs]
        cmd = ["ffmpeg", "-loglevel", "error", "-i", os.path.join(a.src, n + ".mp4"), "-vf", f"fps={a.fps}",
               "-frames:v", str(n_t), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        k = 0
        while k < n_t:
            buf = pr.stdout.read(W0 * H0 * 3)
            if len(buf) < W0 * H0 * 3:
                break
            img = np.frombuffer(buf, np.uint8).reshape(H0, W0, 3)
            for mx, my, d, p in maps:
                u = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(d, p % k), u, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames[split].append({"file_path": f"./{split}/{n}_t{k:03d}", "time": round(k / (n_t - 1), 6),
                                  "transform_matrix": M.tolist()})
            k += 1
        pr.stdout.close()
        pr.wait()
        print(f"{n}: {k} frames -> {split}", flush=True)
    for split in ("train", "test"):
        json.dump({"camera_angle_x": fovx, "w": Wt, "h": Ht, "frames": frames[split]},
                  open(os.path.join(a.out, f"transforms_{split}.json"), "w"), indent=1)
    json.dump({"camera_angle_x": fovx, "frames": []}, open(os.path.join(a.out, "transforms_val.json"), "w"))
    cams_out = {n: {"R": world[n][0].tolist(), "C": world[n][1].tolist()} for n in names}
    meta = {"seq": "imvid_opera", "clip_start": 0.0, "duration": 5.0, "fps": a.fps, "n_times": n_t,
            "fov_x": fovx, "w": Wt, "h": Ht, "K": Kt.tolist(), "mpu_colmap": s,
            "norm": {"X": X.tolist(), "Rw": Rw.tolist(), "scale": s}, "cameras": cams_out,
            "time_mapping": "4DGS time = t_seconds / ((n_times-1)/fps)", "coords": "metres, z up, +x towards cameras"}
    json.dump(meta, open(os.path.join(a.out, "braindance_meta.json"), "w"), indent=1)
    print(f"PREP_DONE {len(frames['train'])} train / {len(frames['test'])} test, {Wt}x{Ht}, fovx {math.degrees(fovx):.1f}")


if __name__ == "__main__":
    main()

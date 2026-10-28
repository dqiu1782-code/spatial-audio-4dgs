"""COLMAP helpers for scenes with known camera poses (used by imvid_colmap.py). Run directly, it prepares
an N3DV (DyNeRF) scene for 4DGaussians, same steps as the repo's preprocess_dynerf.py + colmap.sh + downsample_point.py, without bash/open3d:
  1. frames: camXX.mp4 -> camXX/images/0000.png ... at 1352x1014 (LANCZOS, as the 4DGS loader does)
  2. COLMAP with the known poses (poses_bounds.npy, same conversion as scripts/llff2colmap.py):
     features + exhaustive matching, point_triangulator, then dense MVS (undistort, patch match, fusion)
  3. voxel-downsample the fused cloud to <= 40k points -> points3D_downsample2.ply
  python colmap_known_poses.py E:\\4dgs_data\\dynerf\\cut_roasted_beef [--colmap E:\\hetero\\colmap_env\\Library\\bin\\colmap.exe]
"""
import argparse
import concurrent.futures as cf
import glob
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

W, H = 1352, 1014


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def extract(mp4):
    import cv2
    from PIL import Image
    out = os.path.join(mp4[:-4], "images")
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.isdir(out) and len(os.listdir(out)) >= n - 1:
        return os.path.basename(mp4), len(os.listdir(out)), "kept"
    os.makedirs(out, exist_ok=True)
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)
        img.save(os.path.join(out, "%04d.png" % k), compress_level=1)
        k += 1
    return os.path.basename(mp4), k, "extracted"


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0], [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, v = np.linalg.eigh(K)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


class Colmap:
    def __init__(self, exe, logf):
        self.exe, self.logf = exe, logf
        self.env = dict(os.environ)
        b = os.path.dirname(exe)
        root = os.path.dirname(os.path.dirname(b))
        self.env["PATH"] = os.pathsep.join([b, os.path.join(root, "Library", "bin"), os.path.join(root, "Scripts"),
                                            root, self.env.get("PATH", "")])
        self.help = {}

    def has(self, cmd, opt):
        if cmd not in self.help:
            r = subprocess.run([self.exe, cmd, "-h"], capture_output=True, text=True, env=self.env)
            self.help[cmd] = r.stdout + r.stderr
        return ("--" + opt) in self.help[cmd]

    def run(self, cmd, *args):
        full = [self.exe, cmd] + [str(x) for x in args]
        t0 = time.time()
        with open(self.logf, "a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(full) + "\n")
            f.flush()
            r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, env=self.env)
        if r.returncode != 0:
            raise RuntimeError(f"colmap {cmd} failed ({r.returncode}), see {self.logf}")
        log(f"colmap {cmd}: {time.time() - t0:.0f} s")


def known_pose_model(scene, cm):
    pb = np.load(os.path.join(scene, "poses_bounds.npy"))
    poses = pb[:, :-2].reshape([-1, 3, 5])
    vids = sorted(glob.glob(os.path.join(scene, "cam[0-9][0-9].mp4")))
    assert len(vids) == len(poses), (len(vids), len(poses))
    focal = poses[0, 2, 4] / 2                                  # poses are for 2704x2028
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    cdir = os.path.join(scene, "colmap")
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
    idir = os.path.join(cdir, "images")
    os.makedirs(idir)
    names = []
    for i, v in enumerate(vids):
        names.append("r_%03d.png" % i)
        shutil.copy(os.path.join(v[:-4], "images", "0000.png"), os.path.join(idir, names[-1]))
    db = os.path.join(cdir, "database.db")
    ex = ["--database_path", db, "--image_path", idir, "--ImageReader.single_camera", 1,
          "--ImageReader.camera_model", "SIMPLE_PINHOLE", "--ImageReader.camera_params", f"{focal},{W / 2},{H / 2}"]
    for o in ("SiftExtraction.max_num_features", "FeatureExtraction.max_num_features"):
        if cm.has("feature_extractor", o):
            ex += ["--" + o, 16384]
            break
    cm.run("feature_extractor", *ex)
    cm.run("exhaustive_matcher", "--database_path", db)
    con = sqlite3.connect(db)
    ids = dict(con.execute("SELECT name, image_id FROM images").fetchall())
    cam_id = con.execute("SELECT camera_id FROM cameras").fetchone()[0]
    con.close()
    sp = os.path.join(cdir, "sparse_custom")
    os.makedirs(sp)
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write(f"{cam_id} SIMPLE_PINHOLE {W} {H} {focal} {W / 2} {H / 2}\n")
    with open(os.path.join(sp, "images.txt"), "w") as f:
        for name, pose in zip(names, poses):
            R = -pose[:3, :3].copy()                            # as scripts/llff2colmap.py
            R[:, 0] = -R[:, 0]
            Rw = np.linalg.inv(R)
            T = -Rw @ pose[:3, 3]
            q = rotmat2qvec(Rw)
            f.write(f"{ids[name]} {' '.join(map(str, q))} {' '.join(map(str, T))} {cam_id} {name}\n\n")
    open(os.path.join(sp, "points3D.txt"), "w").close()
    s0 = os.path.join(cdir, "sparse", "0")
    os.makedirs(s0)
    cm.run("point_triangulator", "--database_path", db, "--image_path", idir, "--input_path", sp, "--output_path", s0)
    ws = os.path.join(cdir, "dense", "workspace")
    os.makedirs(ws)
    cm.run("image_undistorter", "--image_path", idir, "--input_path", s0, "--output_path", ws)
    cm.run("patch_match_stereo", "--workspace_path", ws)
    fused = os.path.join(ws, "fused.ply")
    cm.run("stereo_fusion", "--workspace_path", ws, "--output_path", fused)
    return fused


def downsample(src, dst, target=40000):
    from plyfile import PlyData, PlyElement
    v = PlyData.read(src)["vertex"]
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    C = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
    log(f"fused points: {len(X)}")
    vox = 0.02
    while len(X) > target:                                      # as scripts/downsample_point.py
        key = np.floor(X / vox).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        X = np.stack([np.bincount(inv, X[:, k]) for k in range(3)], 1) / cnt[:, None]
        C = np.stack([np.bincount(inv, C[:, k]) for k in range(3)], 1) / cnt[:, None]
        log(f"voxel {vox:.2f}: {len(X)} points")
        vox += 0.01
    dt = [(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")]
    el = np.empty(len(X), dtype=dt)
    el["x"], el["y"], el["z"] = X.T
    el["nx"] = el["ny"] = el["nz"] = 0
    el["red"], el["green"], el["blue"] = np.clip(np.round(C), 0, 255).astype(np.uint8).T
    PlyData([PlyElement.describe(el, "vertex")]).write(dst)
    log(f"-> {dst} ({len(X)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--colmap", default=r"E:\hetero\colmap_env\Library\bin\colmap.exe")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    vids = sorted(glob.glob(os.path.join(a.scene, "cam[0-9][0-9].mp4")))
    log(f"{len(vids)} videos")
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, n, how in ex.map(extract, vids):
            log(f"{name}: {n} frames ({how})")
    cm = Colmap(a.colmap, os.path.join(a.scene, "colmap_log.txt"))
    fused = known_pose_model(a.scene, cm)
    downsample(fused, os.path.join(a.scene, "points3D_downsample2.ply"))
    log("N3DV_PREP_DONE")


if __name__ == "__main__":
    sys.exit(main())

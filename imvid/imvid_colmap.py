"""Windows side of the ImViD pipeline: JPG -> PNG frames, then a dense init cloud from the known poses
(COLMAP: features, exhaustive matching, point_triangulator, patch match, fusion on frame 0), voxel
downsampled to <= 120k points -> fused.ply (read by the 4DGaussians Blender loader).

  python imvid_colmap.py E:\\4dgs_data\\imvid_opera [--colmap E:\\hetero\\colmap_env\\Library\\bin\\colmap.exe]
"""
import argparse
import glob
import json
import os
import shutil
import sqlite3
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_known_poses import Colmap, log, rotmat2qvec  # noqa: E402


def to_png(data):
    js = sorted(glob.glob(os.path.join(data, "train", "*.jpg")) + glob.glob(os.path.join(data, "test", "*.jpg")))
    for i, p in enumerate(js):
        q = p[:-4] + ".png"
        if not os.path.isfile(q):
            Image.open(p).save(q, compress_level=1)
        if i % 500 == 0:
            log(f"png {i}/{len(js)}")


def downsample(src, dst, target):
    from plyfile import PlyData, PlyElement
    v = PlyData.read(src)["vertex"]
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    Cc = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
    log(f"fused points: {len(X)}")
    vox = 0.01
    while len(X) > target:
        key = np.floor(X / vox).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        X = np.stack([np.bincount(inv, X[:, k]) for k in range(3)], 1) / cnt[:, None]
        Cc = np.stack([np.bincount(inv, Cc[:, k]) for k in range(3)], 1) / cnt[:, None]
        vox *= 1.3
    dt = [(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")]
    el = np.zeros(len(X), dtype=dt)
    el["x"], el["y"], el["z"] = X.T
    el["red"], el["green"], el["blue"] = np.clip(np.round(Cc), 0, 255).astype(np.uint8).T
    PlyData([PlyElement.describe(el, "vertex")]).write(dst)
    log(f"-> {dst} ({len(X)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--colmap", default=r"E:\hetero\colmap_env\Library\bin\colmap.exe")
    ap.add_argument("--points", type=int, default=120000)
    a = ap.parse_args()
    to_png(a.data)
    meta = json.load(open(os.path.join(a.data, "scene_meta.json")))
    W, H = meta["w"], meta["h"]
    K = np.array(meta["K"])
    frames = []
    for split in ("train", "test"):
        T = json.load(open(os.path.join(a.data, f"transforms_{split}.json")))
        frames += [f for f in T["frames"] if f["file_path"].endswith("_t000")]
    cdir = os.path.join(a.data, "colmap")
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
    idir = os.path.join(cdir, "images")
    os.makedirs(idir)
    poses = {}
    for f in frames:
        name = os.path.basename(f["file_path"]).replace("_t000", "") + ".png"
        shutil.copy(os.path.join(a.data, f["file_path"][2:] + ".png"), os.path.join(idir, name))
        M = np.array(f["transform_matrix"])
        Rcv = M[:3, :3] @ np.diag([1.0, -1.0, -1.0])               # OpenGL c2w -> OpenCV c2w
        R = Rcv.T
        poses[name] = (R, -R @ M[:3, 3])
    cm = Colmap(a.colmap, os.path.join(a.data, "colmap_log.txt"))
    db = os.path.join(cdir, "database.db")
    params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
    cm.run("feature_extractor", "--database_path", db, "--image_path", idir, "--ImageReader.single_camera", 1,
           "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", params)
    cm.run("exhaustive_matcher", "--database_path", db)
    con = sqlite3.connect(db)
    ids = dict(con.execute("SELECT name, image_id FROM images").fetchall())
    cam_id = con.execute("SELECT camera_id FROM cameras").fetchone()[0]
    con.close()
    sp = os.path.join(cdir, "sparse_custom")
    os.makedirs(sp)
    with open(os.path.join(sp, "cameras.txt"), "w") as g:
        g.write(f"{cam_id} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    with open(os.path.join(sp, "images.txt"), "w") as g:
        for name, (R, t) in poses.items():
            g.write(f"{ids[name]} {' '.join(map(str, rotmat2qvec(R)))} {' '.join(map(str, t))} {cam_id} {name}\n\n")
    open(os.path.join(sp, "points3D.txt"), "w").close()
    s0 = os.path.join(cdir, "sparse", "0")
    os.makedirs(s0)
    cm.run("point_triangulator", "--database_path", db, "--image_path", idir, "--input_path", sp, "--output_path", s0)
    ws = os.path.join(cdir, "dense")
    cm.run("image_undistorter", "--image_path", idir, "--input_path", s0, "--output_path", ws)
    cm.run("patch_match_stereo", "--workspace_path", ws)
    fused = os.path.join(ws, "fused.ply")
    cm.run("stereo_fusion", "--workspace_path", ws, "--output_path", fused)
    downsample(fused, os.path.join(a.data, "fused.ply"), a.points)
    log("IMVID_COLMAP_DONE")


if __name__ == "__main__":
    main()

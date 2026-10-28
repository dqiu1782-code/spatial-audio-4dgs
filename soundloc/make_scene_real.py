"""
make_scene_real.py -- turn real recordings into a scene directory that run_localize.py can read

Input is a config json (see real_config_example.json):
  - per camera: video or wav file, mic layout ("stereo"/"tetra"/"hex" or your own measured coordinates, camera frame, metres),
    pose (the matching image name in COLMAP, or pos + R directly)
  - colmap: sparse/0 directory (camera poses; scene bounds from the point-cloud bounding box)
  - sync: "auto" = align cameras in time with a clap at the start; or a manual offset per camera (seconds)
Audio is extracted from video with ffmpeg (bundled with imageio-ffmpeg).

  python make_scene_real.py real_config.json --out real1
  python run_localize.py real1
Note: COLMAP scale is arbitrary (not metres). Mic coordinates are still in metres (only used for direction, which is scale-free);
     the triangulated result is in the same frame as 3DGS/4DGS. Smoothing parameters are in metres, so set "scale" in the config
     (metres per COLMAP unit; measure any known length in the scene).
"""
import argparse
import os
import subprocess

import numpy as np

from geom import ARRAYS, Camera, load_json, read_wav, save_json, write_wav



def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_images(path):
    import struct
    out = {}
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d, = struct.unpack("<Q", f.read(8))
            f.seek(n2d * 24, 1)
            Rw2c = qvec2rotmat([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            out[name.decode()] = (-Rw2c.T @ t, Rw2c.T)          # camera centre, camera->world rotation
    return out


def colmap_bounds(path):
    import struct
    P = []
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)
            P.append(struct.unpack("<ddd", f.read(24)))
            f.read(3 + 8)
            tl, = struct.unpack("<Q", f.read(8))
            f.seek(8 * tl, 1)
    P = np.array(P)
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    pad = 0.2 * (hi - lo)
    return (lo - pad).tolist(), (hi + pad).tolist()


def extract_audio(src, dst, fs, ch):
    if src.lower().endswith(".wav"):
        x, f2 = read_wav(src)
        if f2 != fs:
            x = np.stack([np.interp(np.arange(int(len(x) * fs / f2)) / fs, np.arange(len(x)) / f2, x[:, k])
                          for k in range(x.shape[1])], 1)
        write_wav(dst, x, fs)
        return
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", str(ch), "-ar", str(fs),
                    "-acodec", "pcm_s16le", dst], check=True)


def auto_sync(X, fs, search=2.0):
    """Estimate each camera's time offset relative to the first by cross-correlating sound envelopes over the first `search` seconds (a clap / beep works best)"""
    names = list(X)
    def env(x):
        e = np.abs(x[: int(search * 3 * fs), 0])
        k = int(0.002 * fs)
        return np.convolve(e, np.ones(k) / k, "same")
    ref = env(X[names[0]])
    off = {names[0]: 0.0}
    for nm in names[1:]:
        e = env(X[nm])
        m = min(len(ref), len(e))
        cc = np.correlate(e[:m] - e[:m].mean(), ref[:m] - ref[:m].mean(), "full")
        lag = np.argmax(cc) - (m - 1)
        off[nm] = float(-lag / fs)   # positive = this camera started recording later
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=int, default=48000)
    a = ap.parse_args()
    cfg = load_json(a.config)
    base = os.path.dirname(os.path.abspath(a.config))
    os.makedirs(a.out, exist_ok=True)
    poses = {}
    if cfg.get("colmap"):
        sp = os.path.join(base, cfg["colmap"])
        poses = read_colmap_images(os.path.join(sp, "images.bin"))
    cams, wavs, X = [], {}, {}
    for c in cfg["cameras"]:
        mics = ARRAYS[c["array"]] if isinstance(c["array"], str) else np.array(c["array"], float)
        if "image_name" in c:
            pos, R = poses[c["image_name"]]
        else:
            pos, R = np.array(c["pos"], float), np.array(c["R"], float)
        cams.append(Camera(c["name"], pos, R, mics))
        dst = os.path.join(a.out, f"{c['name']}.wav")
        extract_audio(os.path.join(base, c["media"]), dst, a.fs, len(mics))
        wavs[c["name"]] = f"{c['name']}.wav"
        X[c["name"]] = read_wav(dst)[0]
    off = auto_sync(X, a.fs) if cfg.get("sync", "auto") == "auto" else cfg["sync"]
    scene = {"fs": a.fs, "cameras": [c.to_dict() for c in cams], "wav": wavs, "time_offset": off,
             "scale": cfg.get("scale", 1.0)}
    if cfg.get("colmap"):
        scene["bounds"] = colmap_bounds(os.path.join(base, cfg["colmap"], "points3D.bin"))
    save_json(os.path.join(a.out, "scene.json"), scene)
    print("time offsets (s):", {k: round(v, 4) for k, v in off.items()})
    print(f"scene written to {a.out}/, next run: python run_localize.py {a.out}")


if __name__ == "__main__":
    main()

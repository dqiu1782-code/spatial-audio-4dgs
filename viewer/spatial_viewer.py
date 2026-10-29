"""
Spatial-audio 4DGS viewer: interactive 4DGS playback with listener-relative spatial audio.

Video: 4DGaussians background + optional sound-anchored dynamic object layer (dynobj.pt),
       rendered on the GPU for the current camera pose and time.
Audio: localized source trajectory + dry signal, binaurally rendered for the current camera
       (propagation delay w/ Doppler, Woodworth ITD, Brown-Duda head shadow, 1/r, rear low-pass).
Clock: the audio device is the master clock; falls back to wall clock without audio.

Usage:
  python spatial_viewer.py --model output/my_scene --configs arguments/dnerf/my_scene.py \
      --sound E:/4dgs_data/my_scene/sound --data E:/4dgs_data/my_scene [--hrtf E:/replay/kemar]

With --hrtf, audio uses measured KEMAR HRIRs + early reflections + late reverb (spatial_hrtf.py);
multi-source scenes (result.json "sources", e.g. Replay actors) require it.

Controls:
  LMB drag  look          W/A/S/D  move          Q/E  down/up        Shift  fast
  Space     play/pause    Left/Right  seek -/+0.5 s                  Home   restart
  1-9       preset camera  M  source markers  T  cycle trajectory  R  reset view
  H         toggle help    Esc  quit
"""
import argparse
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))   # spatial_hrtf

C_SOUND = 343.0   # m/s
HEAD_R = 0.0875   # m


# ---------------------------------------------------------------- camera helpers
def look_dir(yaw, pitch):
    return np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])


def cv_rotation(fwd):
    """Forward vector -> OpenCV camera-to-world rotation (cols: right, down, forward), world z-up."""
    f = fwd / np.linalg.norm(fwd)
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], 1)


class FourDGS:
    """Loads a 4DGaussians model (plus optional dynamic layer) and renders arbitrary camera/time."""

    def __init__(self, root, model_dir, configs, iteration=-1, convention="blender", white_bg=True, use_dyn=True):
        sys.path.insert(0, root)
        import torch
        from argparse import ArgumentParser
        from arguments import ModelHiddenParams, ModelParams, PipelineParams
        from gaussian_renderer import render
        from scene.cameras import MiniCam
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import fov2focal, focal2fov, getProjectionMatrix, getWorld2View2
        from utils.params_utils import merge_hparams
        import mmcv
        self.torch, self._render, self.MiniCam = torch, render, MiniCam
        self.getW2V, self.getProj, self.fov2focal, self.focal2fov = getWorld2View2, getProjectionMatrix, fov2focal, focal2fov
        parser = ArgumentParser()
        mp, pp, hp = ModelParams(parser), PipelineParams(parser), ModelHiddenParams(parser)
        args = parser.parse_args([])
        if configs:
            args = merge_hparams(args, mmcv.Config.fromfile(configs))
        self.pipe, hyper = pp.extract(args), hp.extract(args)
        pc_dir = os.path.join(model_dir, "point_cloud")
        if iteration < 0:
            its = [int(os.path.basename(p).split("_")[1]) for p in glob.glob(os.path.join(pc_dir, "iteration_*"))]
            iteration = max(its)
        it_dir = os.path.join(pc_dir, f"iteration_{iteration}")
        with open(os.path.join(it_dir, "point_cloud.ply"), "rb") as f:      # SH degree of the saved model
            hdr = f.read(4096).split(b"end_header")[0].decode("ascii", "ignore")
        n_rest = hdr.count("f_rest_")
        sh = int(round(((n_rest // 3) + 1) ** 0.5)) - 1 if n_rest else args.sh_degree
        self.g = GaussianModel(sh, hyper)
        self.g._deformation.deformation_net.set_aabb(np.array([1.3, 1.3, 1.3]), np.array([-1.3, -1.3, -1.3]))
        self.g.load_ply(os.path.join(it_dir, "point_cloud.ply"))
        self.g.load_model(it_dir)
        self.g._deformation.eval()
        self.bg = torch.tensor([1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0], device="cuda")
        self.convention = convention
        self.iteration = iteration
        self.n_points = self.g.get_xyz.shape[0]
        # Optional sound-anchored dynamic object layer (from train_dynobj.py).
        self.dyn = None
        dp = os.path.join(model_dir, "dynobj.pt")
        if use_dyn and os.path.isfile(dp):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import dynobj
            self._dynmod = dynobj
            self.dyn = dynobj.DynObj(state=torch.load(dp, map_location="cpu"))
            self.dyn.requires_grad_(False)

    def camera(self, R_cv, C, W, H, fovx, t_norm):
        """R_cv: OpenCV camera-to-world rotation; C: camera centre (training world frame)."""
        torch = self.torch
        if self.convention == "blender":
            # Same conversion as 4DGaussians readCamerasFromTransforms (via OpenGL c2w).
            c2w = np.eye(4)
            c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = R_cv[:, 0], -R_cv[:, 1], -R_cv[:, 2], C
            m = np.linalg.inv(c2w)
            R = -np.transpose(m[:3, :3])
            R[:, 0] = -R[:, 0]
            T = m[:3, 3] * np.array([1.0, -1.0, -1.0])  # matches patched reader
        else:  # colmap: R = cam-to-world rotation, T = world-to-cam translation
            R = R_cv.copy()
            T = -R_cv.T @ C
        wv = torch.tensor(self.getW2V(R, T, np.zeros(3), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
        fovy = self.focal2fov(self.fov2focal(fovx, W), H)
        proj = self.getProj(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        return self.MiniCam(W, H, fovy, fovx, 0.01, 100.0, wv, full, float(t_norm))

    def render(self, cam):
        with self.torch.no_grad():
            if self.dyn is not None:
                D = self._dynmod
                img = D.rasterize(cam, [D.background(self.g, cam.time), self.dyn.at(cam.time)],
                                  self.bg, self.g.active_sh_degree)
            else:
                img = self._render(cam, self.g, self.pipe, self.bg, stage="fine", cam_type=None)["render"]
            return (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().contiguous().cpu().numpy()


# ---------------------------------------------------------------- real-time binaural audio
class SpatialAudio:
    """Block-wise renderer driven by the audio callback. Listener pose is interpolated per block."""

    def __init__(self, dry, fs, traj_t, traj_p, duration, block=512, volume=0.8):
        from scipy.signal import lfilter
        self.lfilter = lfilter
        self.dry = np.asarray(dry, np.float64)
        self.fs, self.block, self.volume = fs, block, volume
        self.duration = duration
        self.set_traj(traj_t, traj_p)
        self.pos = 0                       # playback position in samples (master clock)
        self.playing = True
        self.L = np.zeros(3)
        self.right = np.array([0.0, -1.0, 0.0])
        self.fwd = np.array([1.0, 0.0, 0.0])
        self._prev = None
        self._zi = [np.zeros(1), np.zeros(1)]
        self._lp = [np.zeros(1), np.zeros(1)]
        self.lock = threading.Lock()
        self.stream = None
        self.status = "no audio device"

    def set_traj(self, t, p):
        self.tk, self.pk = np.asarray(t, float), np.asarray(p, float)

    def src(self, ts):
        return np.stack([np.interp(ts, self.tk, self.pk[:, k]) for k in range(3)], -1)

    def time(self):
        return self.pos / self.fs

    def seek(self, t):
        with self.lock:
            self.pos = int(max(0.0, t % self.duration) * self.fs)

    def set_listener(self, L, right, fwd):
        self.L, self.right, self.fwd = np.array(L, float), np.array(right, float), np.array(fwd, float)

    @staticmethod
    def ear_delay(th):
        """Woodworth ITD for incidence angle th (rad) relative to the ear axis."""
        return np.where(th < np.pi / 2, -HEAD_R / C_SOUND * np.cos(th), HEAD_R / C_SOUND * (th - np.pi / 2))

    def _shelf(self, th_mid):
        """Brown-Duda first-order head-shadow filter, bilinear transform."""
        w0 = C_SOUND / HEAD_R
        alpha = 1.05 + 0.95 * math.cos(math.degrees(th_mid) / 150.0 * math.pi)
        K = 2 * self.fs
        b0, b1 = 1 + alpha * K / (2 * w0), 1 - alpha * K / (2 * w0)
        a0, a1 = 1 + K / (2 * w0), 1 - K / (2 * w0)
        return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])

    def block_render(self, n):
        with self.lock:
            p0 = self.pos
            if self.playing:
                self.pos = (self.pos + n) % int(self.duration * self.fs)
        if not self.playing:
            return np.zeros((n, 2), np.float32)
        ts = (p0 + np.arange(n)) / self.fs
        cur = (self.L.copy(), self.right.copy(), self.fwd.copy())
        prev = self._prev or cur
        self._prev = cur
        w = (np.arange(n) / n)[:, None]
        L = prev[0] + (cur[0] - prev[0]) * w
        rgt = prev[1] + (cur[1] - prev[1]) * w
        fwd = prev[2] + (cur[2] - prev[2]) * w
        S = self.src(ts)
        v = S - L
        r = np.maximum(np.linalg.norm(v, axis=1), 0.3)
        u = v / r[:, None]
        behind = np.clip(-np.sum(u * fwd, 1) * 2.0, 0, 1)
        out = np.zeros((n, 2))
        N = len(self.dry)
        for e, sgn in enumerate((-1.0, 1.0)):            # 0 = left, 1 = right
            ear = sgn * rgt / np.linalg.norm(rgt, axis=1, keepdims=True)
            th = np.arccos(np.clip(np.sum(u * ear, 1), -1, 1))
            idx = (ts - (r / C_SOUND + self.ear_delay(th))) * self.fs
            idx = np.mod(idx, N - 1)
            i0 = np.floor(idx).astype(int)
            fr = idx - i0
            x = (self.dry[i0] * (1 - fr) + self.dry[np.minimum(i0 + 1, N - 1)] * fr) / r
            b, a = self._shelf(float(th[n // 2]))
            x, self._zi[e] = self.lfilter(b, a, x, zi=self._zi[e])
            # Rear cue: blend in a ~5 kHz one-pole low-pass.
            c = math.exp(-2 * math.pi * 5000 / self.fs)
            lp, self._lp[e] = self.lfilter([1 - c], [1, -c], x, zi=self._lp[e])
            out[:, e] = x * (1 - behind) + lp * behind
        return np.tanh(self.volume * out).astype(np.float32)

    def start(self):
        try:
            import sounddevice as sd
        except Exception as ex:               # no backend: video still runs on wall clock
            self.status = f"no audio ({ex.__class__.__name__})"
            return False

        def cb(outdata, frames, t, status):
            outdata[:] = self.block_render(frames)

        try:
            self.stream = sd.OutputStream(samplerate=self.fs, channels=2, dtype="float32", blocksize=self.block,
                                          latency="low", callback=cb)
            self.stream.start()
            self.status = f"audio: {sd.query_devices(sd.default.device[1])['name'][:28]}"
            return True
        except Exception as ex:
            self.status = f"no audio ({ex})"[:40]
            return False


# ---------------------------------------------------------------- main
TRAJ_NAME = {"est": "audio only (red)", "fused": "audio + vision (blue)", "truth": "ground truth (grey)"}


def load_multi(sound_dir, track="auto"):
    """Real captures: one source per near-range track, trimmed to the training clip.
    track: 'audio' (audio-only localisation), 'video' (triangulated from the cameras) or 'auto'
    (video when the held-out audio error exceeds 30 cm)."""
    import soundfile as sf
    S = json.load(open(os.path.join(sound_dir, "scene.json"), encoding="utf-8"))
    R = json.load(open(os.path.join(sound_dir, "result.json"), encoding="utf-8"))
    t0, dur = S["clip_start"], S["duration"]
    err = (R.get("heldout_err_cm") or {}).get("median_3d") or 1e9
    srcs = []
    for s in R["sources"]:
        use_video = "pos_video" in s and (track == "video" or (track == "auto" and err > 30))
        key = "pos_video" if use_video else "pos"
        x, fs = sf.read(os.path.join(S["src_dir"], s["dry"]), always_2d=True)
        ta = t0 + R.get("audio_offset_s", 0.0)          # audio-local start of the clip
        x = x.mean(1)[int(ta * fs): int((ta + dur) * fs)]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.05
        srcs.append((s["name"] + (" [video]" if use_video else " [audio]"), x, np.array(s["t"]), np.array(s[key])))
    # Real binaural recordings at each camera, same clip, for A/B listening (key B).
    recs = []
    for c in S.get("cameras", []):
        y, fs2 = sf.read(os.path.join(S["src_dir"], c["binaural"]), always_2d=True)
        ta = t0 + R.get("audio_offset_s", 0.0)
        y = y[int(ta * fs2): int((ta + dur) * fs2), :2]
        recs.append((c["name"], np.array(c["pos"]), y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.08))
    R["_recordings"] = recs
    return S, R, srcs, fs, dur


def load_sound(sound_dir, use):
    from_json = lambda p: json.load(open(p, encoding="utf-8"))   # noqa: E731
    S = from_json(os.path.join(sound_dir, "scene.json"))
    R = from_json(os.path.join(sound_dir, "result.json")) if os.path.isfile(os.path.join(sound_dir, "result.json")) else None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wave

    def read_wav(p):
        with wave.open(p, "rb") as w:
            fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
            x = np.frombuffer(w.readframes(n), "<i2").astype(np.float64).reshape(-1, ch) / 32768.0
        return x, fs
    dry_p = os.path.join(sound_dir, "source_dry.wav")
    if os.path.isfile(dry_p):
        dry, fs = read_wav(dry_p)
        dry = dry.mean(1)
    else:  # real captures have no dry track: use mic 0 of the camera closest to the source on average
        sm = np.array(R["smooth"])
        best = min(S["cameras"], key=lambda c: np.median(np.linalg.norm(sm - np.array(c["pos"]), axis=1)))
        dry, fs = read_wav(os.path.join(sound_dir, S["wav"][best["name"]]))
        dry = dry[:, 0]
    trajs = {}
    if R is not None:
        trajs["est"] = (np.array(R["t"]), np.array(R["smooth"]))
    fp = os.path.join(sound_dir, "result_fused.json")
    if os.path.isfile(fp):  # audio-initialized, multi-view refined trajectory
        F = from_json(fp)
        trajs["fused"] = (np.array(F["t"]), np.array(F["smooth"]))
    if "truth" in S:
        trajs["truth"] = (np.array(S["truth"]["t"]), np.array(S["truth"]["pos"]))
    if use not in trajs:
        use = next(iter(trajs))
    duration = S.get("duration", len(dry) / fs)
    return S, dry, fs, trajs, use, duration


def main():
    ap = argparse.ArgumentParser(description="spatial-audio 4DGS viewer")
    ap.add_argument("--root", default=r"E:\4DGaussians-master", help="4DGaussians source directory")
    ap.add_argument("--model", required=True, help="training output dir (contains point_cloud/iteration_*)")
    ap.add_argument("--configs", default="", help="config file used for training")
    ap.add_argument("--sound", required=True, help="soundloc scene dir (scene.json, result.json, wav)")
    ap.add_argument("--data", default="", help="training data dir (transforms_train.json gives preset cameras)")
    ap.add_argument("--use", default="fused", choices=["fused", "est", "truth"],
                    help="trajectory driving the audio: fused (default if present), est, or truth")
    ap.add_argument("--no_dyn", action="store_true", help="disable the dynamic object layer")
    ap.add_argument("--hrtf", default="", help="KEMAR HRIR directory (enables HRTF + room rendering)")
    ap.add_argument("--reverb", type=float, default=0.18, help="late reverb level (HRTF mode)")
    ap.add_argument("--track", default="auto", choices=["auto", "audio", "video"],
                    help="multi-source scenes: source positions from audio localisation or from video")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--convention", default="blender", choices=["blender", "colmap"])
    ap.add_argument("--volume", type=float, default=0.8)
    ap.add_argument("--iteration", type=int, default=-1)
    a = ap.parse_args()

    import pygame
    multi = False
    R_json = {}
    rp = os.path.join(a.sound, "result.json")
    if os.path.isfile(rp):
        R_json = json.load(open(rp, encoding="utf-8"))
        multi = "sources" in R_json
    if multi:
        if not a.hrtf:
            raise SystemExit("multi-source scene: pass --hrtf <KEMAR dir>")
        S, R_json, msrc, fs, duration = load_multi(a.sound, a.track)
        trajs = {name: (t, p) for name, _, t, p in msrc}
        use = msrc[0][0]
        dry = None
    else:
        S, dry, fs, trajs, use, duration = load_sound(a.sound, a.use)
    print(f"Loading 4DGS model: {a.model}", flush=True)
    gs = FourDGS(a.root, a.model, a.configs, a.iteration, a.convention, use_dyn=not a.no_dyn)
    print(f"  {gs.n_points} Gaussians, iteration {gs.iteration}"
          + (f", dynamic layer: {gs.dyn.xyz.shape[0]} Gaussians" if gs.dyn is not None else ", no dynamic layer"),
          flush=True)
    W = H = a.res
    meta = {}
    meta_p = os.path.join(a.data, "scene_meta.json") if a.data else ""
    if meta_p and os.path.isfile(meta_p):
        meta = json.load(open(meta_p, encoding="utf-8"))
        if "w" in meta:
            W, H = int(meta["w"]), int(meta["h"])
    fovx = math.radians(a.fov)
    fx = 0.5 * W / math.tan(0.5 * fovx)

    presets = []
    if a.data and os.path.isfile(os.path.join(a.data, "transforms_train.json")):
        tj = json.load(open(os.path.join(a.data, "transforms_train.json")))
        fovx = tj.get("camera_angle_x", fovx)
        fx = 0.5 * W / math.tan(0.5 * fovx)
        seen = set()
        for f in tj["frames"]:
            M = np.array(f["transform_matrix"])
            key = tuple(np.round(M[:3, 3], 3))
            if key not in seen:
                seen.add(key)
                presets.append(M.copy())
    if not presets:
        presets = [np.array(c["pos"]) for c in S.get("cameras", [])]
    # Model time: frame k has time k/(n-1) at k/fps seconds, so t_norm = seconds / time_span.
    time_span = duration
    if meta:
        time_span = (meta["n_times"] - 1) / meta["fps"]
    target = np.mean([p.mean(0) for _, p in trajs.values()], 0) if multi else np.array([0.0, 0.0, 0.3])

    def pose_from(p):
        if np.ndim(p) == 2:                        # training camera (OpenGL c2w): keep its view direction
            f = -p[:3, 2] / np.linalg.norm(p[:3, 2])
            return p[:3, 3].copy(), math.atan2(f[1], f[0]), math.asin(np.clip(f[2], -1, 1))
        f = target - p
        f /= np.linalg.norm(f)
        return p.copy(), math.atan2(f[1], f[0]), math.asin(np.clip(f[2], -1, 1))

    pos, yaw, pitch = pose_from(presets[0])
    zone, edge = None, False                     # captured volume: (centre, r_ok, r_max), world units
    if meta.get("view_zone"):
        vz = meta["view_zone"]
        zone = (np.array(vz["center"], float), float(vz["r_ok"]), float(vz["r_max"]))
        pos, yaw, pitch = pose_from(zone[0].copy())
    home = (pos.copy(), yaw, pitch)

    if a.hrtf:
        import spatial_hrtf as SH
        hrtf = SH.HRTFSet(a.hrtf, fs)
        mpu = float(R_json.get("metres_per_unit", 1.0))
        if multi:
            sources = [SH.Source(x, t, p, name) for name, x, t, p in msrc]
            # Approximate room: box around the camera ring, floor ~1.5 m below camera height.
            room = (np.array([-1.25, -1.25, -1.5 / mpu]), np.array([1.25, 1.25, 1.3 / mpu]))
            if "room" in S:                    # scene-specific box (world units)
                room = (np.array(S["room"][0], float), np.array(S["room"][1], float))
        else:
            sources = [SH.Source(dry, *trajs[use])]
            room = (np.array([-1.6, -1.6, 0.0]), np.array([1.6, 1.6, 2.6]))
        audio = SH.HRTFRenderer(sources, fs, duration, hrtf, room=room, reverb_gain=a.reverb,
                                volume=a.volume, metres_per_unit=mpu)
    else:
        audio = SpatialAudio(dry, fs, *trajs[use], duration, volume=a.volume)
    has_audio = audio.start()
    wall_t = 0.0

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    try:
        pygame.key.stop_text_input()                # disables the IME so WASD reach the key state
    except Exception:
        pass
    pygame.display.set_caption("spatial-audio-4dgs | 4DGS + Spatial Audio")
    font = None
    # YaHei UI covers Latin and CJK (device names may be localized).
    for name in ("microsoftyaheiui", "microsoftyahei", "segoeui", None):
        try:
            font = pygame.font.SysFont(name, 18) if name else pygame.font.Font(None, 20)
            break
        except Exception:
            continue
    clock = pygame.time.Clock()
    show_marker, show_help, dragging = True, True, False
    marker_r = 0.15 / float(R_json.get("metres_per_unit", 1.0)) if multi else 0.2   # world units
    recs = R_json.get("_recordings", []) if multi else []
    real_mode, real_name = False, ""
    fps_hist = []
    running = True
    while running:
        dt = clock.tick(120) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    audio.playing = not audio.playing
                elif k == pygame.K_LEFT:
                    audio.seek(audio.time() - 0.5)
                    wall_t = max(0.0, wall_t - 0.5)
                elif k == pygame.K_RIGHT:
                    audio.seek(audio.time() + 0.5)
                    wall_t += 0.5
                elif k == pygame.K_HOME:
                    audio.seek(0.0)
                    wall_t = 0.0
                elif k == pygame.K_m:
                    show_marker = not show_marker
                elif k == pygame.K_b and recs:
                    real_mode = not real_mode
                    if not real_mode:
                        audio.override, real_name = None, ""
                elif k == pygame.K_h:
                    show_help = not show_help
                elif k == pygame.K_t and len(trajs) > 1 and not multi:
                    names = list(trajs)
                    use = names[(names.index(use) + 1) % len(names)]
                    audio.set_traj(*trajs[use])
                elif k == pygame.K_r:
                    pos, yaw, pitch = home[0].copy(), home[1], home[2]
                elif pygame.K_1 <= k <= pygame.K_9 and k - pygame.K_1 < len(presets):
                    pos, yaw, pitch = pose_from(presets[k - pygame.K_1])
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                yaw -= ev.rel[0] * 0.004
                pitch = float(np.clip(pitch - ev.rel[1] * 0.004, -1.45, 1.45))
        keys = pygame.key.get_pressed()
        spd = (3.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0) * dt
        fwd = look_dir(yaw, pitch)
        Rcv = cv_rotation(fwd)
        right = Rcv[:, 0]
        flat = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            pos += flat * spd
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            pos -= flat * spd
        if keys[pygame.K_d]:
            pos += right * spd
        if keys[pygame.K_a]:
            pos -= right * spd
        if keys[pygame.K_e]:
            pos[2] += spd
        if keys[pygame.K_q]:
            pos[2] -= spd
        if zone is not None:
            off = pos - zone[0]
            dz = float(np.linalg.norm(off))
            if dz > zone[2]:
                pos = zone[0] + off * (zone[2] / dz)
            edge = dz > zone[1]
        audio.set_listener(pos, right, fwd)
        if real_mode:
            near = min(recs, key=lambda r: np.linalg.norm(r[1] - pos))
            if near[0] != real_name:
                audio.override, real_name = near[2], near[0]
        if has_audio:
            t = audio.time()
        else:
            if audio.playing:
                wall_t = (wall_t + dt) % duration
            t = wall_t
        t0 = time.perf_counter()
        img = gs.render(gs.camera(Rcv, pos, W, H, fovx, min(t / time_span, 1.0)))
        fps_hist = (fps_hist + [1.0 / max(time.perf_counter() - t0, 1e-4)])[-30:]
        screen.blit(pygame.image.frombuffer(img.tobytes(), (W, H), "RGB"), (0, 0))
        if show_marker:
            palette = [(230, 60, 40), (40, 150, 255), (60, 200, 80), (240, 200, 40)]
            marks = (list(zip(trajs, palette)) if multi else
                     (("truth", (130, 130, 130)), ("est", (230, 60, 40)), ("fused", (40, 150, 255))))
            for name, col in marks:
                if name not in trajs:
                    continue
                tk, pk = trajs[name]
                p = np.array([np.interp(t, tk, pk[:, k]) for k in range(3)])
                q = Rcv.T @ (p - pos)
                if q[2] > 0.05:
                    u, v = fx * q[0] / q[2] + W / 2, fx * q[1] / q[2] + H / 2
                    rad = max(6, int(marker_r * fx / q[2]))
                    pygame.draw.circle(screen, col, (int(u), int(v)), rad, 3 if (multi or name == use) else 1)
        if font:
            hud = [f"t = {t:5.2f} / {duration:.1f} s   {'Playing' if audio.playing else 'Paused'}   "
                   f"{np.mean(fps_hist):.0f} fps",
                   (f"Sources: {', '.join(trajs)}   {audio.status}" if multi else
                    f"Source: {TRAJ_NAME.get(use, use)}  [T]   {audio.status}")]
            if multi and recs:
                hud.append(f"Audio [B]: {'REAL recording @ ' + real_name if real_mode else 'rendered (HRTF + room)'}")
            if edge:
                hud.append("Edge of the captured area: no cameras saw the scene from further out")
            if show_help:
                hud += ["Drag: look  |  WASD: move  |  Q/E: down/up  |  Shift: fast",
                        "Space: pause  |  Left/Right: seek  |  1-9: cameras  |  M: markers  |  R: reset  |  H: help"]
            for i, line in enumerate(hud):
                s = font.render(line, True, (20, 20, 20))
                bgr = pygame.Surface((s.get_width() + 10, s.get_height() + 4), pygame.SRCALPHA)
                bgr.fill((255, 255, 255, 170))
                screen.blit(bgr, (6, 6 + i * 24))
                screen.blit(s, (11, 8 + i * 24))
        if edge:
            pygame.draw.rect(screen, (255, 140, 0), (0, 0, W, H), 6)
        pygame.display.flip()
    if audio.stream is not None:
        audio.stream.stop()
        audio.stream.close()
    pygame.quit()


if __name__ == "__main__":
    main()

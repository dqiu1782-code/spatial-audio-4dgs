"""Capture rig specification (rig.json): what the person recording declares about the setup.

Conventions
-----------
World: metres, z up, origin on the floor at the stage centre. Seen from above, camera ring
angles are measured counter-clockwise from +x, which is the "front" of the stage.
Cameras are numbered 1..N counter-clockwise, starting at the start of the viewing arc
(the front, angle 0, for a full ring).
Viewing arc ("view": {"arc_deg": [start, end]}, default the full ring): the directions the viewer may
watch from. Inside it neighbouring cameras should be <= 15 deg apart (> 20 deg gets blurry), so that
novel views are interpolated between close cameras instead of guessed.
Camera frame (OpenCV): x right, y down, z forward. Mic offsets are given in this frame.

Two tiers
---------
basic    : phones; each records video and its own audio (1-2 mics, axis along the phone's long edge).
advanced : cameras + external mics (stereo pair, binaural head, tetra array) and optional lavaliers.

The declared geometry is a prior: self-calibration (selfcal.py) refines mic offsets, axes and latencies.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np

MIC_LAYOUTS = {
    # name: list of channel positions in the mic's own frame (x = mic axis), metres, before 'spacing' scaling
    "mono": [[0.0, 0.0, 0.0]],
    "stereo": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],          # spacing = distance between capsules
    "binaural": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],        # ears; spacing ~0.16-0.18 m
    "tetra": [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]],
}
DEVICE_DEFAULTS = {
    # offset_sd: how well the mic position on the device is known; drift_sd: clock drift prior
    "phone": {"layout": "stereo", "spacing": 0.14, "latency_sd": 0.002, "offset_sd": 0.02, "drift_sd": 20e-6},
    "gopro": {"layout": "stereo", "spacing": 0.03, "latency_sd": 0.005, "offset_sd": 0.02, "drift_sd": 30e-6},
    "dslr": {"layout": "stereo", "spacing": 0.02, "latency_sd": 0.010, "offset_sd": 0.03, "drift_sd": 30e-6},
    "recorder": {"layout": "stereo", "spacing": 0.17, "latency_sd": 0.050, "offset_sd": 0.05, "drift_sd": 50e-6},
}


VIEW_GAP_OK, VIEW_GAP_MAX = 15.0, 20.0   # deg between neighbouring cameras inside the viewing arc


def view_arc(rig):
    """(start, width, full_ring) of the viewing arc in degrees, counter-clockwise from the front."""
    a = rig.raw.get("view", {}).get("arc_deg", [0.0, 360.0])
    width = float(a[1]) - float(a[0])
    return float(a[0]) % 360, min(width, 360.0), width >= 359.9


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """OpenCV camera-to-world rotation (columns: right, down, forward)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f], 1)


def rot_axis(yaw_deg, pitch_deg):
    """Unit vector in the camera frame for a mic axis: yaw about camera y (down), pitch about x."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.cos(p) * math.cos(y), -math.sin(p), math.cos(p) * math.sin(y)])


@dataclass
class Mic:
    id: str
    device: str
    layout: str
    spacing: float
    offset: np.ndarray            # camera frame, m
    axis_yaw: float               # deg, mic axis relative to camera right, about camera down
    axis_pitch: float
    audio: str
    channels: list
    latency_sd: float             # prior std of the recorder's latency, s
    offset_sd: float = 0.05       # prior std of the mic position on the device, m
    drift_sd: float = 50e-6       # prior std of the recorder clock drift

    def channel_positions_cam(self, offset=None, yaw=None, pitch=None, spacing=None):
        """Channel positions in the camera frame (n_ch, 3)."""
        offset = self.offset if offset is None else offset
        yaw = self.axis_yaw if yaw is None else yaw
        pitch = self.axis_pitch if pitch is None else pitch
        spacing = self.spacing if spacing is None else spacing
        ex = rot_axis(yaw, pitch)
        ey = np.cross(np.array([0.0, 1.0, 0.0]), ex)
        ey = ey / (np.linalg.norm(ey) + 1e-12)
        ez = np.cross(ex, ey)
        L = np.array(MIC_LAYOUTS[self.layout]) * spacing
        return offset + L[:, 0:1] * ex + L[:, 1:2] * ey + L[:, 2:3] * ez


@dataclass
class Camera:
    id: str
    order: int
    device: str
    C: np.ndarray                 # world centre
    R: np.ndarray                 # cam-to-world (OpenCV)
    video: str
    fov_deg: float
    mics: list = field(default_factory=list)


@dataclass
class Rig:
    name: str
    tier: str
    cameras: list
    lavaliers: list
    room: dict
    sync: dict
    room_tone: dict
    raw: dict

    @property
    def mics(self):
        return [m for c in self.cameras for m in c.mics]

    def camera_of(self, mic):
        return next(c for c in self.cameras if mic in c.mics)

    def channel_positions_world(self, mic, **kw):
        c = self.camera_of(mic)
        return c.C + mic.channel_positions_cam(**kw) @ c.R.T


def _camera_from_spec(cs, stage_target):
    a = math.radians(cs["angle_deg"])
    C = np.array([cs["radius_m"] * math.cos(a), cs["radius_m"] * math.sin(a), cs["height_m"]])
    aim = cs.get("aim", {})
    if "target" in aim:
        R = look_at(C, aim["target"])
    else:
        yaw = math.radians(aim.get("yaw_deg", 0.0))          # 0 = facing the stage centre
        pitch = math.radians(aim.get("pitch_deg", 0.0))      # + = tilted up
        to_c = -C[:2] / (np.linalg.norm(C[:2]) + 1e-12)
        h = math.atan2(to_c[1], to_c[0]) + yaw
        f = np.array([math.cos(pitch) * math.cos(h), math.cos(pitch) * math.sin(h), math.sin(pitch)])
        R = look_at(C, C + f)
    if not aim:
        R = look_at(C, stage_target)
    cam = Camera(cs["id"], cs["order"], cs.get("device", "phone"), C, R, cs.get("video", ""),
                 cs.get("fov_deg", 70.0))
    for ms in cs.get("mics", [{}]):
        d = DEVICE_DEFAULTS.get(ms.get("device", cam.device), DEVICE_DEFAULTS["phone"])
        layout = ms.get("layout", d["layout"])
        cam.mics.append(Mic(
            id=ms.get("id", cs["id"] + "_mic"), device=ms.get("device", cam.device),
            layout=layout, spacing=ms.get("spacing_m", d["spacing"]),
            offset=np.array(ms.get("offset_m", [0.0, 0.0, 0.0]), float),
            axis_yaw=ms.get("axis_yaw_deg", 0.0), axis_pitch=ms.get("axis_pitch_deg", 0.0),
            audio=ms.get("audio", cam.video), channels=ms.get("channels", list(range(len(MIC_LAYOUTS[layout])))),
            latency_sd=ms.get("latency_sd_s", d["latency_sd"]),
            offset_sd=ms.get("offset_sd_m", d["offset_sd"]), drift_sd=ms.get("drift_sd", d["drift_sd"])))
    return cam


def load_rig(path_or_dict):
    spec = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict, encoding="utf-8"))
    target = spec.get("stage", {}).get("target", [0.0, 0.0, 1.3])
    cams = sorted((_camera_from_spec(c, target) for c in spec["cameras"]), key=lambda c: c.order)
    return Rig(spec.get("name", "capture"), spec.get("tier", "basic"), cams, spec.get("lavaliers", []),
               spec.get("room", {}), spec.get("sync", {"method": "clap"}), spec.get("room_tone", {}), spec)


def validate(rig):
    """Returns (errors, warnings) for a declared rig: ordering, coverage, heights, sync."""
    err, warn = [], []
    orders = [c.order for c in rig.cameras]
    if sorted(orders) != list(range(1, len(orders) + 1)):
        err.append(f"camera order must be 1..{len(orders)} without gaps, got {sorted(orders)}")
    if len(rig.cameras) < 4:
        err.append("at least 4 cameras are needed for 3D reconstruction around the stage")
    a0, width, full = view_arc(rig)
    ang = [math.degrees(math.atan2(c.C[1], c.C[0])) % 360 for c in rig.cameras]
    u = [(x - a0 + (0.0 if full else 5.0)) % 360 - (0.0 if full else 5.0) for x in ang]   # along the arc
    seq = [u[i] for i in np.argsort(orders)]
    if any(seq[i + 1] <= seq[i] for i in range(len(seq) - 1)):
        warn.append("cameras are not numbered counter-clockwise from the start of the viewing arc")
    vis = sorted(x for x, c in zip(u, rig.cameras) if c.video and -5.0 <= x <= width + 5.0)
    need = int(math.ceil(width / VIEW_GAP_OK)) + (0 if full else 1)
    if len(vis) >= 2:
        gaps = np.diff(vis + [vis[0] + 360]) if full else np.diff(vis)
        g = float(gaps.max())
        if g > VIEW_GAP_MAX + 0.5:
            warn.append(f"largest gap between neighbouring cameras in the viewing arc is {g:.0f} deg (> {VIEW_GAP_MAX:.0f}): "
                        f"novel views there will be blurry; {need} cameras at {VIEW_GAP_OK:.0f} deg cover this arc")
        elif g > VIEW_GAP_OK + 0.5:
            warn.append(f"largest gap between neighbouring cameras in the viewing arc is {g:.0f} deg; "
                        f"<= {VIEW_GAP_OK:.0f} deg is recommended ({need} cameras)")
        if not full and (vis[0] > 5.0 or vis[-1] < width - 5.0):
            warn.append("no camera at one end of the viewing arc: put one at each end or narrow view.arc_deg")
    else:
        warn.append("fewer than 2 cameras inside the viewing arc")
    hs = np.array([c.C[2] for c in rig.cameras])
    if hs.std() < 0.15:
        warn.append("all cameras at about the same height: sound-source height will be poorly determined; "
                    "alternate heights (e.g. 1.1 m / 1.8 m)")
    tgt = np.asarray(rig.raw.get("stage", {}).get("target", [0.0, 0.0, 1.3]), float)
    for c in rig.cameras:
        d = tgt - c.C
        cosang = np.dot(c.R[:, 2], d / np.linalg.norm(d))
        if cosang < math.cos(math.radians(20)):
            warn.append(f"{c.id} does not face the stage target ({math.degrees(math.acos(np.clip(cosang, -1, 1))):.0f} deg off)")
    n_ch = sum(len(MIC_LAYOUTS[m.layout]) for m in rig.mics)
    if n_ch < 4:
        err.append("at least 4 microphone channels are needed for 3D sound localisation")
    if rig.sync.get("method") not in ("clap", "timecode", "shared_recorder"):
        warn.append("sync method should be 'clap' (default), 'timecode' or 'shared_recorder'")
    if not rig.lavaliers:
        warn.append("no lavaliers: voice/ambience separation drops from ~8 dB to ~4 dB; a phone recording in each actor's pocket is enough")
    if not rig.raw.get("empty_plate"):
        warn.append("record ~10 s with the actors off stage (empty plate): it gives a clean static background")
    if not rig.room_tone:
        warn.append("record 10-20 s of silence (room tone) at the start: it becomes the ambience bed")
    return err, warn


def layout_svg(rig, path, size=520):
    """Top-down sketch: cameras (numbered, with view direction and mic axis), stage, room."""
    pts = np.array([c.C[:2] for c in rig.cameras])
    lim = max(np.abs(pts).max() * 1.25, 1.0)
    if rig.room.get("size_m"):
        lim = max(lim, max(rig.room["size_m"][:2]) / 2 * 1.05)
    s = size / (2 * lim)
    X = lambda v: size / 2 + v[0] * s          # noqa: E731
    Y = lambda v: size / 2 - v[1] * s          # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" font-family="sans-serif">',
           '<rect width="100%" height="100%" fill="#fafafa"/>']
    if rig.room.get("size_m"):
        Lx, Ly = rig.room["size_m"][:2]
        out.append(f'<rect x="{X([-Lx / 2, 0]):.1f}" y="{Y([0, Ly / 2]):.1f}" width="{Lx * s:.1f}" '
                   f'height="{Ly * s:.1f}" fill="none" stroke="#999" stroke-dasharray="6 4"/>')
    a0, width, full = view_arc(rig)
    if not full:                                   # viewing arc as a wedge
        R = lim * 0.95
        p0 = [R * math.cos(math.radians(a0)), R * math.sin(math.radians(a0))]
        p1 = [R * math.cos(math.radians(a0 + width)), R * math.sin(math.radians(a0 + width))]
        out.append(f'<path d="M {X([0, 0])} {Y([0, 0])} L {X(p0):.1f} {Y(p0):.1f} A {R * s:.1f} {R * s:.1f} 0 '
                   f'{int(width > 180)} 0 {X(p1):.1f} {Y(p1):.1f} Z" fill="#fff6d8" stroke="#e0c060"/>')
        out.append(f'<rect x="10" y="10" width="14" height="10" fill="#fff6d8" stroke="#e0c060"/>'
                   f'<text x="30" y="19" font-size="11" fill="#a80">viewing arc {width:.0f} deg</text>')
    out.append(f'<circle cx="{X([0, 0])}" cy="{Y([0, 0])}" r="{0.8 * s:.1f}" fill="#e8f0ff" stroke="#8aa"/>')
    out.append(f'<text x="{X([0, 0])}" y="{Y([0, 0]) + 4}" font-size="12" text-anchor="middle" fill="#557">stage</text>')
    out.append(f'<text x="{X([0.85, 0]):.1f}" y="{Y([0, 0]) + 4:.1f}" font-size="11" fill="#c33">front &#8594;</text>')
    for c in rig.cameras:
        p, f = c.C[:2], c.R[:2, 2] / (np.linalg.norm(c.R[:2, 2]) + 1e-12)
        q = p + f * 0.45
        out.append(f'<line x1="{X(p):.1f}" y1="{Y(p):.1f}" x2="{X(q):.1f}" y2="{Y(q):.1f}" stroke="#333" stroke-width="2"/>')
        out.append(f'<circle cx="{X(p):.1f}" cy="{Y(p):.1f}" r="11" fill="#fff" stroke="#333" stroke-width="2"/>')
        out.append(f'<text x="{X(p):.1f}" y="{Y(p) + 4:.1f}" font-size="12" text-anchor="middle">{c.order}</text>')
        o = p / (np.linalg.norm(p) + 1e-12) * (26 / s)          # label just outside the ring
        out.append(f'<text x="{X(p + o):.1f}" y="{Y(p + o) + 3:.1f}" font-size="9" text-anchor="middle" fill="#666">'
                   f'{c.C[2]:.1f}m</text>')
        for m in c.mics:
            P = rig.channel_positions_world(m)
            if len(P) >= 2:
                a, b = P[0, :2], P[-1, :2]
                mid, d = (a + b) / 2, (b - a) / (np.linalg.norm(b - a) + 1e-12) * 0.18
                out.append(f'<line x1="{X(mid - d):.1f}" y1="{Y(mid - d):.1f}" x2="{X(mid + d):.1f}" '
                           f'y2="{Y(mid + d):.1f}" stroke="#d55" stroke-width="2"/>')
    out.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    import sys
    r = load_rig(sys.argv[1])
    e, w = validate(r)
    print(f"{r.name}: {len(r.cameras)} cameras, {sum(len(MIC_LAYOUTS[m.layout]) for m in r.mics)} mic channels, "
          f"{len(r.lavaliers)} lavaliers")
    for x in e:
        print("ERROR  ", x)
    for x in w:
        print("WARNING", x)
    if len(sys.argv) > 2:
        layout_svg(r, sys.argv[2])
        print("layout ->", sys.argv[2])

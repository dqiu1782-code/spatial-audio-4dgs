"""geom.py -- camera poses, microphone array geometry, wav I/O (numpy and standard library only)"""
import json
import wave

import numpy as np

C_SOUND = 343.0          # speed of sound, m/s


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation R (columns = camera x right, y down, z forward), same as COLMAP/OpenCV camera coordinates"""
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


# microphone arrays (camera coordinates, metres): x right, y down, z forward
ARRAYS = {
    # two mics along the camera's left-right axis, 15 cm apart (like a clip-on stereo mic / the two ends of a phone)
    "stereo": np.array([[-0.075, 0, 0], [0.075, 0, 0]]),
    # four mics on a regular tetrahedron, radius 5 cm (like a small ambisonic mic); gives a full 3D direction
    "tetra": 0.05 / np.sqrt(3) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
    # six-mic ring plus one above and one below, radius 4 cm
    "hex": np.array([[0.04 * np.cos(a), 0.0, 0.04 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
                    + [[0, -0.03, 0], [0, 0.03, 0]]),
}


class Camera:
    def __init__(self, name, pos, R, mics_local):
        self.name = name
        self.pos = np.asarray(pos, float)
        self.R = np.asarray(R, float)               # camera -> world
        self.mics_local = np.asarray(mics_local, float)

    @property
    def mics_world(self):
        return self.pos + self.mics_local @ self.R.T

    def to_dict(self):
        return {"name": self.name, "pos": self.pos.tolist(), "R": self.R.tolist(), "mics_local": self.mics_local.tolist()}

    @staticmethod
    def from_dict(d):
        return Camera(d["name"], d["pos"], d["R"], d["mics_local"])


def write_wav(path, x, fs):
    """x: (n,) or (n, ch) float in +-1 -> 16-bit wav"""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.clip(x, -1, 1)
    y = (y * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(y.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(y.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        fs, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float64) / (1 << 23)
    else:
        raise ValueError(f"unsupported {8 * sw}-bit wav")
    return x.reshape(-1, ch), fs


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

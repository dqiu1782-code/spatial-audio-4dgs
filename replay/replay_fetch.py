"""Fetch one Replay scene (8 DSLR videos + all audio) with parallel HTTP range reads, plus MIT KEMAR HRTFs.

Only the needed members are read out of the large zip archives (stored members are copied byte-for-byte).

  python replay_fetch.py SC-1044
"""
import concurrent.futures as cf
import os
import struct
import sys
import time
import urllib.request as u
import zipfile
import zlib

from remotezip import RemoteZip

BASE = "https://dl.fbaipublicfiles.com/replay/v0/"
CHUNKS = "1001-1007 1008-1018 1019-1028 1032-1044 1045-1052 1053-1071 1073-1079 1081-1091 1092-1104 1105-1109"
PART = 8 << 20
THREADS = 12
seq = sys.argv[1] if len(sys.argv) > 1 else "SC-1044"
num = int(seq.split("-")[1])
out = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), seq)
os.makedirs(out, exist_ok=True)


def chunk_for(n):
    for c in CHUNKS.split():
        a, b = map(int, c.split("-"))
        if a <= n <= b:
            return f"videos_SC-{c}.zip"
    raise ValueError(n)


def get_range(url, a, b, tries=6):
    for k in range(tries):
        try:
            req = u.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with u.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except Exception:
            time.sleep(1 + k)
    raise IOError(f"range {a}-{b} failed")


def fetch_member(url, info, dst):
    hdr = get_range(url, info.header_offset, info.header_offset + 29)
    n_name, n_extra = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + n_name + n_extra
    size = info.compress_size
    parts = [(o, min(o + PART, size) - 1) for o in range(0, size, PART)]
    t0 = time.time()
    with open(dst + ".part", "wb") as f:
        f.truncate(size)
        with cf.ThreadPoolExecutor(THREADS) as ex:
            futs = {ex.submit(get_range, url, start + a, start + b): a for a, b in parts}
            for fu in cf.as_completed(futs):
                f.seek(futs[fu])
                f.write(fu.result())
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = open(dst + ".part", "rb").read()
        open(dst, "wb").write(zlib.decompress(raw, -15))
        os.remove(dst + ".part")
    else:
        os.replace(dst + ".part", dst)
    mb = size / 2 ** 20
    print(f"  {os.path.basename(dst):28s} {mb:7.1f} MB  {mb / max(time.time() - t0, 1e-3):5.1f} MB/s", flush=True)


def pull(arc, want):
    url = BASE + arc
    with RemoteZip(url) as z:
        infos = [i for i in z.infolist() if want(i.filename)]
    for info in infos:
        dst = os.path.join(out, os.path.basename(info.filename))
        if os.path.isfile(dst) and os.path.getsize(dst) == info.file_size:
            print(f"  skip {os.path.basename(dst)}")
            continue
        fetch_member(url, info, dst)


print("audio ...", flush=True)
pull("audio_SC-1001-1109.zip", lambda n: n.startswith(f"audio/{seq}/") and n.endswith(".flac"))
print("video ...", flush=True)
pull(chunk_for(num), lambda n: n.startswith(f"videos/{seq}/DSLR-") and n.endswith(".mkv"))

hr = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "kemar")
os.makedirs(hr, exist_ok=True)
zp = os.path.join(hr, "diffuse.zip")
if not os.path.isfile(zp):
    u.urlretrieve("http://sound.media.mit.edu/resources/KEMAR/diffuse.zip", zp)
with zipfile.ZipFile(zp) as z:
    z.extractall(hr)
    print("kemar entries", len(z.namelist()), z.namelist()[:5])
print("FETCH_DONE")

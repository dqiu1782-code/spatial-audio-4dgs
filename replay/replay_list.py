"""Download Replay metadata and list archive contents without downloading the archives."""
import os
import shutil
import sys
import urllib.request as u
import zipfile

from remotezip import RemoteZip

BASE = "https://dl.fbaipublicfiles.com/replay/v0/"
OUT = os.environ.get("REPLAY_ROOT", r"E:\replay")
os.makedirs(OUT, exist_ok=True)
print("E: free GB", round(shutil.disk_usage("E:\\").free / 2 ** 30, 1))

meta = os.path.join(OUT, "metadata.zip")
if not os.path.isfile(meta):
    u.urlretrieve(BASE + "metadata.zip", meta)
with zipfile.ZipFile(meta) as z:
    names = z.namelist()
    z.extractall(os.path.join(OUT, "metadata"))
print("metadata entries", len(names))
for n in names[:40]:
    print("  ", n)

with open(os.path.join(OUT, "listing.txt"), "w", encoding="utf-8") as f:
    for arc in ("audio_SC-1001-1109.zip", "videos_SC-1001-1007.zip"):
        with RemoteZip(BASE + arc) as z:
            info = z.infolist()
        f.write(f"## {arc} {len(info)} entries\n")
        for i in info:
            f.write(f"{i.file_size / 2 ** 20:10.1f} MB  {i.filename}\n")
        print(arc, len(info), "entries")
print("LIST_DONE")

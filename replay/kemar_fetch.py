"""Fetch the MIT KEMAR diffuse-field HRIR set (221 KB) with an explicit CA bundle."""
import os
import ssl
import urllib.request as u
import zipfile

try:
    import certifi
    ca = certifi.where()
except ImportError:
    from pip._vendor import certifi
    ca = certifi.where()
out = os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "kemar")
os.makedirs(out, exist_ok=True)
zp = os.path.join(out, "diffuse.zip")
ctx = ssl.create_default_context(cafile=ca)
with u.urlopen("https://sound.media.mit.edu/resources/KEMAR/diffuse.zip", context=ctx, timeout=60) as r:
    open(zp, "wb").write(r.read())
with zipfile.ZipFile(zp) as z:
    z.extractall(out)
    names = z.namelist()
print("kemar entries", len(names), names[:4])
print("KEMAR_DONE")

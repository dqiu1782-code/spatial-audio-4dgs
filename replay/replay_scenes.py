"""Summarize Replay scenes: scenario, actors, duration, audio sensors."""
import os
import sqlite3

db = sqlite3.connect(os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "metadata", "metadata.sqlite"))
rows = db.execute("SELECT sequence_name, scenario, space_id FROM sequence_annots ORDER BY sequence_name").fetchall()
out = []
for seq, scen, space in rows:
    t0, t1, nf = db.execute("SELECT MIN(frame_timestamp), MAX(frame_timestamp), COUNT(*) FROM frame_annots "
                            "WHERE sequence_name=? AND sensor_name='DSLR-1'", (seq,)).fetchone()
    actors = [a for (a,) in db.execute("SELECT DISTINCT actor_id FROM actor_frame_annots WHERE sequence_name=?", (seq,))]
    sens = db.execute("SELECT sensor_name, sensor_type, timestamp_delta FROM sensor_annots WHERE sequence_name=?",
                      (seq,)).fetchall()
    aud = [s for s in sens if s[1] != "video"]
    out.append(f"{seq} space={space} dur={0 if t0 is None else t1 - t0:6.1f}s frames={nf} actors={actors} "
               f"audio={len(aud)} | {scen[:90]}")
seq = "SC-1044"
out.append("\n## sensors " + seq)
for r in db.execute("SELECT * FROM sensor_annots WHERE sequence_name=?", (seq,)):
    out.append("  " + str(r))
out.append("\n## frame sensors " + seq)
for r in db.execute("SELECT sensor_name, COUNT(*), MIN(frame_timestamp), MAX(frame_timestamp) FROM frame_annots "
                    "WHERE sequence_name=? GROUP BY sensor_name", (seq,)):
    out.append("  " + str(r))
open(os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "scenes.txt"), "w", encoding="utf-8").write("\n".join(out))
print("SCENES_DONE")

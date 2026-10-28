"""Dump the Replay metadata.sqlite schema and sample rows."""
import os
import sqlite3

db = sqlite3.connect(os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "metadata", "metadata.sqlite"))
out = []
for (name, sql) in db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
    n = db.execute(f"SELECT COUNT(*) FROM '{name}'").fetchone()[0]
    out.append(f"## {name} rows={n}\n{sql}")
    cur = db.execute(f"SELECT * FROM '{name}' LIMIT 2")
    cols = [d[0] for d in cur.description]
    for row in cur.fetchall():
        out.append("  " + " | ".join(f"{c}={str(v)[:300]}" for c, v in zip(cols, row)))
open(os.path.join(os.environ.get("REPLAY_ROOT", r"E:\replay"), "schema.txt"), "w", encoding="utf-8").write("\n".join(out))
print("SCHEMA_DONE")

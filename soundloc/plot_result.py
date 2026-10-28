"""plot_result.py -- plot localisation results: top view (cameras, rays, ground truth, estimated track) + error over time"""
import os

import numpy as np


def plot(scene_dir, S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    CAM, RAY, GT, RAW, SM = "#2a78d6", "#86b6ef", "#898781", "#eda100", "#c0392b"
    t = np.array(out["t"])
    raw = np.array([r if r is not None else [np.nan] * 3 for r in out["raw"]], float)
    sm = np.array(out["smooth"])
    valid = np.array(out["valid"])
    gt = np.array(out["gt"]) if "gt" in out else None

    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    if "room" in S:
        L, W, _ = S["room"]
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color=MUTED, lw=1)
    # pick one frame to draw rays
    k = int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 3]) if valid.any() else 0
    for c in S["cameras"]:
        pos, R = np.array(c["pos"]), np.array(c["R"])
        ax.scatter(pos[0], pos[1], s=110, marker="s", color=CAM, zorder=5)
        f = R[:, 2]
        ax.annotate("", xy=(pos[0] + 0.5 * f[0], pos[1] + 0.5 * f[1]), xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle="->", color=CAM, lw=1.2))
        ax.text(pos[0], pos[1] - 0.3, c["name"], ha="center", fontsize=8.5, color=INK2)
        dd = out["doa"][c["name"]][k] if k < len(out["doa"][c["name"]]) else None
        if dd and dd["kind"] == "dir":
            d = np.array(dd["d"])
            ax.plot([pos[0], pos[0] + 8 * d[0]], [pos[1], pos[1] + 8 * d[1]], color=RAY, lw=1.3, zorder=1)
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], color=GT, lw=2.5, label="ground-truth track", zorder=2)
    ax.scatter(raw[valid, 0], raw[valid, 1], s=7, color=RAW, alpha=0.6, label="per-frame triangulation", zorder=3)
    ax.plot(sm[:, 0], sm[:, 1], color=SM, lw=1.6, label="smoothed track", zorder=4)
    if valid.any():
        ax.scatter(raw[k, 0], raw[k, 1], s=60, color=RAW, edgecolor=INK, zorder=6, label=f"rays and intersection at t={t[k]:.2f}s")
    ax.set_aspect("equal")
    if "room" in S:
        ax.set_xlim(-0.5, S["room"][0] + 0.5)
        ax.set_ylim(-0.5, S["room"][1] + 0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ttl = "Top view: sound direction measured by each camera -> triangulation"
    if "eval" in out:
        ttl += f"\nmedian per-frame error {out['eval']['raw_median_m']:.2f} m, smoothed {out['eval']['smooth_median_m']:.2f} m"
    ax.set_title(ttl, fontsize=10.5, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(scene_dir, "fig_topview.png"), dpi=150)
    plt.close(fig)

    if gt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), facecolor=SURF, sharex=True)
        e_raw = np.linalg.norm(raw - gt, axis=1)
        e_sm = np.linalg.norm(sm - gt, axis=1)
        axes[0].plot(t[valid], e_raw[valid], ".", ms=3, color=RAW, label="per frame")
        axes[0].plot(t, e_sm, color=SM, lw=1.4, label="smoothed")
        axes[0].set_ylabel("localisation error (m)", color=INK2)
        axes[0].legend(fontsize=8, frameon=False)
        for d, lab in zip(range(3), "xyz"):
            axes[1].plot(t, gt[:, d], color=GT, lw=2)
            axes[1].plot(t, sm[:, d], lw=1.2, label=lab)
        axes[1].set_ylabel("coordinate (m)", color=INK2)
        axes[1].set_xlabel("time (s)", color=INK2)
        axes[1].legend(fontsize=8, frameon=False, ncol=3)
        for a in axes:
            a.set_facecolor(SURF)
            a.grid(True, color=GRID, lw=0.7)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(scene_dir, "fig_error.png"), dpi=150)
        plt.close(fig)
    print(f"figures saved: {scene_dir}/fig_topview.png" + (f", fig_error.png" if gt is not None else ""))

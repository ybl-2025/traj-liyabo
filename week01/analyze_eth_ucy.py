from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = {
    "seq_eth": ("ETH/seq_eth/obsmat.txt", 15.0),
    "seq_hotel": ("ETH/seq_hotel/obsmat.txt", 15.0),
    "zara01": ("UCY/zara01/obsmat.txt", 25.0),
    "zara02": ("UCY/zara02/obsmat.txt", 25.0),
    "students03": ("UCY/students03/obsmat.txt", 25.0),
}

COLS = ["frame", "ped_id", "x", "z", "y", "vx", "vz", "vy"]
COLLISION_DISTANCE_M = 0.6
TTC_HORIZON_S = 10.0


def load_data(path: Path, raw_fps: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", names=COLS, engine="python")
    df = df[["frame", "ped_id", "x", "y", "vx", "vy"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["ped_id"] = df["ped_id"].astype(int)
    df["time_s"] = (df["frame"] - df["frame"].min()) / raw_fps
    df["speed"] = np.hypot(df["vx"], df["vy"])
    df["direction_deg"] = (np.degrees(np.arctan2(df["vy"], df["vx"])) + 360) % 360
    df = df.sort_values(["ped_id", "time_s"]).reset_index(drop=True)
    dt = df.groupby("ped_id")["time_s"].diff()
    dvx = df.groupby("ped_id")["vx"].diff()
    dvy = df.groupby("ped_id")["vy"].diff()
    good = dt.gt(0) & dt.le(2.0)
    df["ax"] = np.where(good, dvx / dt, np.nan)
    df["ay"] = np.where(good, dvy / dt, np.nan)
    df["acceleration"] = np.hypot(df["ax"], df["ay"])
    return df


def interaction_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["nearest_distance"] = np.nan
    df["ttc"] = np.nan
    pairs = []
    for frame, idx in df.groupby("frame", sort=True).groups.items():
        ids = np.asarray(list(idx))
        block = df.loc[ids]
        p = block[["x", "y"]].to_numpy(float)
        v = block[["vx", "vy"]].to_numpy(float)
        ped = block["ped_id"].to_numpy(int)
        n = len(block)
        if n < 2:
            continue
        delta = p[None, :, :] - p[:, None, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        df.loc[ids, "nearest_distance"] = dist.min(axis=1)
        min_ttc = np.full(n, np.inf)
        for i in range(n - 1):
            for j in range(i + 1, n):
                r = p[j] - p[i]
                u = v[j] - v[i]
                d = float(np.linalg.norm(r))
                a = float(np.dot(u, u))
                b = 2.0 * float(np.dot(r, u))
                c = float(np.dot(r, r) - COLLISION_DISTANCE_M**2)
                ttc = np.nan
                if a > 1e-10:
                    disc = b * b - 4 * a * c
                    if disc >= 0:
                        root = (-b - math.sqrt(disc)) / (2 * a)
                        if 0 <= root <= TTC_HORIZON_S:
                            ttc = root
                            min_ttc[i] = min(min_ttc[i], root)
                            min_ttc[j] = min(min_ttc[j], root)
                h1 = math.atan2(v[i, 1], v[i, 0])
                h2 = math.atan2(v[j, 1], v[j, 0])
                hdiff = abs(math.atan2(math.sin(h1 - h2), math.cos(h1 - h2)))
                mean_v = (v[i] + v[j]) / 2
                mean_speed = float(np.linalg.norm(mean_v))
                lateral_ratio = 0.0
                if d > 1e-9 and mean_speed > 1e-9:
                    lateral_ratio = abs(np.cross(mean_v / mean_speed, r)) / d
                formation = 0.5 <= d <= 2.5 and hdiff <= math.radians(20) and lateral_ratio >= 0.7
                if d <= 5.0 or np.isfinite(ttc) or formation:
                    pairs.append(
                        {
                            "frame": int(frame),
                            "time_s": float(block["time_s"].iloc[0]),
                            "ped1": int(ped[i]),
                            "ped2": int(ped[j]),
                            "distance": d,
                            "heading_diff_deg": math.degrees(hdiff),
                            "lateral_ratio": lateral_ratio,
                            "formation": formation,
                            "ttc": ttc,
                        }
                    )
        df.loc[ids, "ttc"] = np.where(np.isfinite(min_ttc), min_ttc, np.nan)
    return df, pd.DataFrame(pairs)


def scene_frame_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    xlo, xhi = np.quantile(df["x"], [0.01, 0.99])
    ylo, yhi = np.quantile(df["y"], [0.01, 0.99])
    area = max(float((xhi - xlo) * (yhi - ylo)), 1.0)
    frames = (
        df.groupby("frame")
        .agg(time_s=("time_s", "first"), pedestrians=("ped_id", "nunique"), mean_speed=("speed", "mean"))
        .reset_index()
    )
    frames["density"] = frames["pedestrians"] / area
    frames["flow"] = frames["density"] * frames["mean_speed"]
    return frames, area


def save_a1(df: pd.DataFrame, out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for _, g in df.groupby("ped_id"):
        ax.plot(g["x"], g["y"], lw=0.65, alpha=0.55)
    ax.set(title=f"A1 Trajectories — {name}", xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.savefig(out / "A1_trajectories.png", dpi=180)
    plt.close(fig)


def save_a2(df: pd.DataFrame, out: Path, name: str) -> None:
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax1 = fig.add_subplot(131)
    ax1.hist(df["speed"].clip(upper=df["speed"].quantile(0.995)), bins=40, color="#2878B5")
    ax1.set(xlabel="speed (m/s)", ylabel="observations", title="Speed")
    ax2 = fig.add_subplot(132, projection="polar")
    angles = np.radians(df["direction_deg"])
    ax2.hist(angles, bins=24, color="#F39B7F", alpha=0.9)
    ax2.set_title("Direction", pad=18)
    ax3 = fig.add_subplot(133)
    acc = df["acceleration"].dropna()
    acc = acc[acc <= acc.quantile(0.99)]
    ax3.hist(acc, bins=40, color="#3C5488")
    ax3.set(xlabel="acceleration magnitude (m/s²)", ylabel="observations", title="Acceleration")
    fig.suptitle(f"A2 Kinematics — {name}")
    fig.savefig(out / "A2_kinematics.png", dpi=180)
    plt.close(fig)


def save_a3(df: pd.DataFrame, out: Path, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    nd = df["nearest_distance"].dropna()
    axes[0].hist(nd[nd <= nd.quantile(0.99)], bins=40, color="#00A087")
    axes[0].set(xlabel="nearest-neighbor distance (m)", ylabel="observations", title="Spacing distribution")
    ttc = df["ttc"].dropna()
    if len(ttc):
        axes[1].hist(ttc, bins=np.linspace(0, TTC_HORIZON_S, 31), color="#E64B35")
    else:
        axes[1].text(0.5, 0.5, "No TTC events", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="TTC (s)", ylabel="observations", title=f"TTC (distance threshold {COLLISION_DISTANCE_M} m)")
    fig.suptitle(f"A3 Pair interactions — {name}")
    fig.savefig(out / "A3_distance_TTC.png", dpi=180)
    plt.close(fig)


def save_a4(frames: pd.DataFrame, out: Path, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].scatter(frames["density"], frames["mean_speed"], s=10, alpha=0.35, color="#4DBBD5")
    axes[0].set(xlabel="density k (ped/m²)", ylabel="mean speed v (m/s)", title="k–v")
    axes[1].scatter(frames["density"], frames["flow"], s=10, alpha=0.35, color="#E64B35")
    axes[1].set(xlabel="density k (ped/m²)", ylabel="flow q (ped/m/s)", title="q–k")
    axes[2].scatter(frames["mean_speed"], frames["flow"], s=10, alpha=0.35, color="#00A087")
    axes[2].set(xlabel="mean speed v (m/s)", ylabel="flow q (ped/m/s)", title="q–v")
    fig.suptitle(f"A4 Fundamental diagrams — {name}")
    fig.savefig(out / "A4_q_k_v.png", dpi=180)
    plt.close(fig)


def pick_pair(pairs: pd.DataFrame, mask: pd.Series, rank_col: str, ascending: bool) -> tuple[int, int] | None:
    x = pairs.loc[mask].copy()
    if x.empty:
        return None
    counts = x.groupby(["ped1", "ped2"]).agg(count=("frame", "size"), score=(rank_col, "min" if ascending else "max"))
    counts = counts.sort_values(["count", "score"], ascending=[False, ascending])
    return tuple(map(int, counts.index[0]))


def plot_pair_paths(ax, df: pd.DataFrame, pair: tuple[int, int] | None, title: str, center_time: float | None = None) -> None:
    if pair is None:
        ax.text(0.5, 0.5, "No event found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    sub = df[df["ped_id"].isin(pair)]
    if center_time is not None:
        sub = sub[sub["time_s"].between(center_time - 4, center_time + 4)]
    for pid, g in sub.groupby("ped_id"):
        ax.plot(g["x"], g["y"], marker=".", ms=2, lw=1.2, label=f"ped {pid}")
    ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7)


def save_a5(df: pd.DataFrame, pairs: pd.DataFrame, out: Path, name: str) -> dict:
    formation_pair = None if pairs.empty else pick_pair(pairs, pairs["formation"], "lateral_ratio", False)
    ttc_mask = pairs["ttc"].notna() & (pairs["ttc"] <= 5) if not pairs.empty else pd.Series(dtype=bool)
    avoidance_pair = None if pairs.empty else pick_pair(pairs, ttc_mask, "ttc", True)
    center_time = None
    if avoidance_pair is not None:
        q = pairs[(pairs["ped1"] == avoidance_pair[0]) & (pairs["ped2"] == avoidance_pair[1]) & pairs["ttc"].notna()]
        if len(q):
            center_time = float(q.loc[q["ttc"].idxmin(), "time_s"])

    osc_scores = []
    for pid, g in df.groupby("ped_id"):
        if len(g) < 12:
            continue
        h = np.unwrap(np.arctan2(g["vy"].to_numpy(), g["vx"].to_numpy()))
        dh = np.diff(h)
        turns = int(np.sum(np.sign(dh[1:]) != np.sign(dh[:-1])))
        osc_scores.append((turns, len(g), int(pid)))
    osc_pid = max(osc_scores, default=(0, 0, -1))[2]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    plot_pair_paths(axes[0, 0], df, formation_pair, "Walking abreast candidate")
    plot_pair_paths(axes[0, 1], df, avoidance_pair, "Avoidance candidate", center_time)
    if osc_pid >= 0:
        g = df[df["ped_id"] == osc_pid]
        axes[1, 0].plot(g["x"], g["y"], color="#3C5488", lw=1.3)
        axes[1, 0].scatter(g["x"].iloc[[0, -1]], g["y"].iloc[[0, -1]], c=["green", "red"], s=35)
        axes[1, 0].set(title=f"Oscillation candidate — ped {osc_pid}", xlabel="x (m)", ylabel="y (m)")
        axes[1, 0].set_aspect("equal", adjustable="datalim")
    hb = axes[1, 1].hexbin(df["x"], df["y"], C=df["speed"], reduce_C_function=np.mean, gridsize=24, mincnt=2, cmap="viridis_r")
    axes[1, 1].set(title="Bottleneck clue: mean speed by location", xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(hb, ax=axes[1, 1], label="mean speed (m/s)")
    fig.suptitle(f"A5 Behavior diagnostics — {name}")
    fig.savefig(out / "A5_behavior.png", dpi=180)
    plt.close(fig)
    return {
        "formation_pair": "" if formation_pair is None else f"{formation_pair[0]}-{formation_pair[1]}",
        "avoidance_pair": "" if avoidance_pair is None else f"{avoidance_pair[0]}-{avoidance_pair[1]}",
        "oscillation_ped": "" if osc_pid < 0 else osc_pid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, (relative, raw_fps) in DATASETS.items():
        source = args.data_root / relative
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"Processing {name}: {source}", flush=True)
        df = load_data(source, raw_fps)
        df, pairs = interaction_metrics(df)
        frames, area = scene_frame_metrics(df)
        save_a1(df, out, name)
        save_a2(df, out, name)
        save_a3(df, out, name)
        save_a4(frames, out, name)
        behavior = save_a5(df, pairs, out, name)
        df.to_csv(out / "trajectory_metrics.csv", index=False, float_format="%.6f")
        pairs.to_csv(out / "pair_events.csv", index=False, float_format="%.6f")
        frames.to_csv(out / "frame_qkv.csv", index=False, float_format="%.6f")
        summaries.append(
            {
                "dataset": name,
                "pedestrians": df["ped_id"].nunique(),
                "observations": len(df),
                "duration_s": df["time_s"].max() - df["time_s"].min(),
                "mean_speed_m_s": df["speed"].mean(),
                "median_nearest_m": df["nearest_distance"].median(),
                "ttc_observations": df["ttc"].notna().sum(),
                "scene_area_m2": area,
                **behavior,
            }
        )
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False, float_format="%.4f")
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()

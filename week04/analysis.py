from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# -----------------------------
# 1. Analysis configuration
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_FILENAME = "NGSIM_I80_20050413_1600-1615_SI.csv"
DATA_CANDIDATES = (
    ROOT / "data" / DATA_FILENAME,
    ROOT.parent / "第4周_NGSIM-I80_课堂数据" / DATA_FILENAME,
)
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
RESULT_DIR = ROOT / "results"

LANE_ID = 4
Y_MIN = 300.0
Y_MAX = 400.0
LENGTH_M = Y_MAX - Y_MIN
DETECTOR_Y = 350.0
TIME_START = 0.0
TIME_END = 900.0
SAMPLE_DT = 0.1
WINDOWS = (10, 60)

COLORS = {
    "edie": "#1769AA",
    "snapshot": "#2A9D8F",
    "detector": "#E76F51",
    "dark": "#23313F",
    "muted": "#617181",
    "grid": "#D9E0E6",
    "accent": "#B33F62",
}

METHODS = ("Edie轨迹裁剪", "0.1秒空间采样", "350米虚拟检测器")
METHOD_COLORS = {
    "Edie轨迹裁剪": COLORS["edie"],
    "0.1秒空间采样": COLORS["snapshot"],
    "350米虚拟检测器": COLORS["detector"],
}


def configure_plotting() -> None:
    """Use a Chinese-capable font and restrained report styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#8896A4",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        searched = "\n".join(f"- {path}" for path in DATA_CANDIDATES)
        raise FileNotFoundError(f"未找到原始数据文件，已检查：\n{searched}")
    columns = [
        "vehicle_id",
        "frame_id",
        "time_s",
        "global_time_ms",
        "local_y_m",
        "lane_id",
        "speed_m_s",
    ]
    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        dtype={
            "vehicle_id": "int32",
            "frame_id": "int32",
            "time_s": "float64",
            "global_time_ms": "int64",
            "local_y_m": "float64",
            "lane_id": "int8",
            "speed_m_s": "float64",
        },
    )
    return df.sort_values(["vehicle_id", "time_s"], kind="mergesort").reset_index(drop=True)


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Build adjacent trajectory segments and append a 0.1 s tail at t=899.9 s.

    The tail closes the final sampling interval [899.9, 900.0). It is only
    added for vehicles still present at the end of the supplied data.
    """
    grouped = df.groupby("vehicle_id", sort=False)
    seg = pd.DataFrame(
        {
            "vehicle_id": df["vehicle_id"],
            "t1": df["time_s"],
            "y1": df["local_y_m"],
            "lane1": df["lane_id"],
            "v1": df["speed_m_s"],
            "t0": grouped["time_s"].shift(),
            "y0": grouped["local_y_m"].shift(),
            "lane0": grouped["lane_id"].shift(),
            "v0": grouped["speed_m_s"].shift(),
        }
    ).dropna()

    final_rows = df[np.isclose(df["time_s"], TIME_END - SAMPLE_DT)].copy()
    tail = pd.DataFrame(
        {
            "vehicle_id": final_rows["vehicle_id"].to_numpy(),
            "t0": final_rows["time_s"].to_numpy(),
            "t1": np.full(len(final_rows), TIME_END),
            "y0": final_rows["local_y_m"].to_numpy(),
            "y1": (
                final_rows["local_y_m"].to_numpy()
                + final_rows["speed_m_s"].to_numpy() * SAMPLE_DT
            ),
            "lane0": final_rows["lane_id"].to_numpy(),
            "lane1": final_rows["lane_id"].to_numpy(),
            "v0": final_rows["speed_m_s"].to_numpy(),
            "v1": final_rows["speed_m_s"].to_numpy(),
        }
    )
    seg = pd.concat([seg, tail], ignore_index=True)
    seg["lane0"] = seg["lane0"].astype("int8")
    seg["lane1"] = seg["lane1"].astype("int8")
    seg["dt"] = seg["t1"] - seg["t0"]
    seg["dy"] = seg["y1"] - seg["y0"]
    return seg


def clip_segments_to_study_area(segments: pd.DataFrame) -> pd.DataFrame:
    """Clip same-lane trajectory segments to y in [Y_MIN, Y_MAX)."""
    valid = segments[
        (segments["lane0"] == LANE_ID)
        & (segments["lane1"] == LANE_ID)
        & (segments["dt"] > 0)
        & (segments["dt"] <= SAMPLE_DT + 1e-7)
        & (segments["dy"] >= -1e-8)
        & (segments["t0"] < TIME_END)
        & (segments["t1"] > TIME_START)
    ].copy()

    dy = valid["dy"].to_numpy()
    y0 = valid["y0"].to_numpy()
    moving = dy > 1e-12

    a_start = np.zeros(len(valid), dtype=float)
    a_end = np.ones(len(valid), dtype=float)
    a_start[moving] = np.maximum(0.0, (Y_MIN - y0[moving]) / dy[moving])
    a_end[moving] = np.minimum(1.0, (Y_MAX - y0[moving]) / dy[moving])

    stationary_inside = (~moving) & (y0 >= Y_MIN) & (y0 < Y_MAX)
    keep = (moving & (a_end > a_start)) | stationary_inside
    clipped = valid.loc[keep].copy()
    a_start = a_start[keep]
    a_end = a_end[keep]

    clipped["clip_t0"] = clipped["t0"].to_numpy() + a_start * clipped["dt"].to_numpy()
    clipped["clip_t1"] = clipped["t0"].to_numpy() + a_end * clipped["dt"].to_numpy()
    clipped["clip_dt"] = clipped["clip_t1"] - clipped["clip_t0"]
    clipped["clip_distance"] = np.maximum(
        0.0, (a_end - a_start) * clipped["dy"].to_numpy()
    )
    clipped["mid_t"] = (clipped["clip_t0"] + clipped["clip_t1"]) / 2.0
    return clipped[clipped["clip_dt"] > 0].copy()


def empty_window_table(window_s: int) -> pd.DataFrame:
    count = int((TIME_END - TIME_START) / window_s)
    result = pd.DataFrame({"window_id": np.arange(count, dtype=int)})
    result["start_s"] = result["window_id"] * window_s
    result["end_s"] = result["start_s"] + window_s
    return result


def calculate_edie(clipped: pd.DataFrame, window_s: int) -> pd.DataFrame:
    data = clipped.copy()
    data["window_id"] = np.floor(data["mid_t"] / window_s).astype(int)
    data = data[(data["window_id"] >= 0) & (data["window_id"] < TIME_END / window_s)]
    grouped = data.groupby("window_id").agg(
        total_time_s=("clip_dt", "sum"),
        total_distance_m=("clip_distance", "sum"),
        segment_count=("vehicle_id", "size"),
        vehicle_count=("vehicle_id", "nunique"),
    )
    result = empty_window_table(window_s).join(grouped, on="window_id").fillna(0)
    result["density_veh_km"] = result["total_time_s"] / (LENGTH_M * window_s) * 1000
    result["flow_veh_h"] = result["total_distance_m"] / (LENGTH_M * window_s) * 3600
    result["speed_km_h"] = np.where(
        result["density_veh_km"] > 0,
        result["flow_veh_h"] / result["density_veh_km"],
        np.nan,
    )
    result["method"] = "Edie轨迹裁剪"
    result["window_s"] = window_s
    return result


def calculate_snapshot(df: pd.DataFrame, window_s: int) -> pd.DataFrame:
    sample = df[
        (df["lane_id"] == LANE_ID)
        & (df["local_y_m"] >= Y_MIN)
        & (df["local_y_m"] < Y_MAX)
        & (df["time_s"] >= TIME_START)
        & (df["time_s"] < TIME_END)
    ].copy()
    sample["window_id"] = np.floor(sample["time_s"] / window_s).astype(int)
    grouped = sample.groupby("window_id").agg(
        record_count=("vehicle_id", "size"),
        vehicle_count=("vehicle_id", "nunique"),
        speed_sum_m_s=("speed_m_s", "sum"),
    )
    result = empty_window_table(window_s).join(grouped, on="window_id").fillna(0)
    result["density_veh_km"] = result["record_count"] * SAMPLE_DT / (LENGTH_M * window_s) * 1000
    result["flow_veh_h"] = result["speed_sum_m_s"] * SAMPLE_DT / (LENGTH_M * window_s) * 3600
    result["speed_km_h"] = np.where(
        result["density_veh_km"] > 0,
        result["flow_veh_h"] / result["density_veh_km"],
        np.nan,
    )
    result["method"] = "0.1秒空间采样"
    result["window_s"] = window_s
    return result


def calculate_detector(segments: pd.DataFrame, window_s: int) -> pd.DataFrame:
    valid = segments[
        (segments["lane0"] == LANE_ID)
        & (segments["lane1"] == LANE_ID)
        & (segments["dt"] > 0)
        & (segments["dt"] <= SAMPLE_DT + 1e-7)
        & (segments["y0"] < DETECTOR_Y)
        & (segments["y1"] >= DETECTOR_Y)
        & (segments["dy"] > 0)
    ].copy()
    alpha = (DETECTOR_Y - valid["y0"]) / valid["dy"]
    valid["cross_time"] = valid["t0"] + alpha * valid["dt"]
    valid["cross_speed_km_h"] = (valid["v0"] + alpha * (valid["v1"] - valid["v0"])) * 3.6
    valid = valid[
        (valid["cross_time"] >= TIME_START)
        & (valid["cross_time"] < TIME_END)
        & (valid["cross_speed_km_h"] > 0)
    ].copy()
    valid["window_id"] = np.floor(valid["cross_time"] / window_s).astype(int)

    def harmonic_mean(values: pd.Series) -> float:
        arr = values.to_numpy(dtype=float)
        return len(arr) / np.sum(1.0 / arr) if len(arr) else np.nan

    grouped = valid.groupby("window_id").agg(
        crossing_count=("vehicle_id", "size"),
        vehicle_count=("vehicle_id", "nunique"),
        speed_km_h=("cross_speed_km_h", harmonic_mean),
    )
    result = empty_window_table(window_s).join(grouped, on="window_id")
    result["crossing_count"] = result["crossing_count"].fillna(0)
    result["vehicle_count"] = result["vehicle_count"].fillna(0)
    result["flow_veh_h"] = result["crossing_count"] / window_s * 3600
    result["density_veh_km"] = np.where(
        result["speed_km_h"] > 0,
        result["flow_veh_h"] / result["speed_km_h"],
        np.nan,
    )
    result["method"] = "350米虚拟检测器"
    result["window_s"] = window_s
    return result


def binned_summary(data: pd.DataFrame, bin_width: float = 10.0, min_count: int = 2) -> pd.DataFrame:
    max_k = math.ceil(data["density_veh_km"].max() / bin_width) * bin_width
    bins = np.arange(0, max_k + bin_width, bin_width)
    work = data.copy()
    work["density_bin"] = pd.cut(work["density_veh_km"], bins=bins, right=False)
    result = work.groupby("density_bin", observed=True).agg(
        k=("density_veh_km", "median"),
        q_med=("flow_veh_h", "median"),
        q25=("flow_veh_h", lambda x: x.quantile(0.25)),
        q75=("flow_veh_h", lambda x: x.quantile(0.75)),
        n=("flow_veh_h", "size"),
    )
    return result[result["n"] >= min_count].reset_index(drop=True)


def common_limits(tables: list[pd.DataFrame], pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    k_max = max(float(t["density_veh_km"].max()) for t in tables)
    q_max = max(float(t["flow_veh_h"].max()) for t in tables)
    x_top = math.ceil(k_max * (1 + pad) / 10) * 10
    y_top = math.ceil(q_max * (1 + pad) / 200) * 200
    return (0.0, x_top), (0.0, y_top)


def style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("密度 k（veh/km/lane）")
    ax.set_ylabel("流量 q（veh/h/lane）")
    ax.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_one_page(all_results: pd.DataFrame, quality: dict) -> None:
    """Create the report figure: three methods by two aggregation windows."""
    tables = [all_results[all_results["window_s"] == w] for w in WINDOWS]
    xlim, ylim = common_limits(tables, pad=0.08)
    speed = all_results["speed_km_h"].replace([np.inf, -np.inf], np.nan).dropna()
    norm = Normalize(vmin=float(speed.min()), vmax=float(speed.max()))

    fig = plt.figure(figsize=(14.5, 10.5), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.78, 6.7, 1.35],
        left=0.075,
        right=0.94,
        top=0.965,
        bottom=0.055,
        hspace=0.22,
    )

    header = fig.add_subplot(outer[0])
    header.axis("off")
    header.text(
        0,
        0.9,
        "NGSIM I-80 单车道流量—密度结果",
        fontsize=19,
        fontweight="bold",
        color=COLORS["dark"],
        va="top",
        transform=header.transAxes,
    )
    header.plot([0, 1], [0.62, 0.62], color="#98A6B3", lw=1.1, transform=header.transAxes)
    header.text(
        0,
        0.38,
        "数据：NGSIM_I80_20050413_1600-1615_SI.csv   方向：I-80东向   时间：16:00–16:15（900 s）",
        fontsize=10.2,
        color=COLORS["dark"],
        transform=header.transAxes,
    )
    header.text(
        0,
        0.08,
        "范围：4号车道，local_y_m ∈ [300, 400) m，L=100 m   窗口：10秒/60秒，左闭右开、非重叠   三种计算口径共用坐标范围",
        fontsize=10.2,
        color=COLORS["dark"],
        transform=header.transAxes,
    )

    plot_grid = outer[1].subgridspec(3, 2, hspace=0.34, wspace=0.18)
    axes = np.empty((3, 2), dtype=object)
    scatter_handle = None
    for row, method in enumerate(METHODS):
        for col, window_s in enumerate(WINDOWS):
            ax = fig.add_subplot(plot_grid[row, col])
            axes[row, col] = ax
            data = all_results[
                (all_results["method"] == method) & (all_results["window_s"] == window_s)
            ]
            scatter_handle = ax.scatter(
                data["density_veh_km"],
                data["flow_veh_h"],
                c=data["speed_km_h"],
                cmap="viridis",
                norm=norm,
                s=25 if window_s == 10 else 48,
                alpha=0.82,
                edgecolors="white" if method != "350米虚拟检测器" else "none",
                linewidths=0.35,
                zorder=3,
            )
            style_axis(ax, xlim, ylim)
            ax.tick_params(labelsize=8.3)
            if row < 2:
                ax.set_xlabel("")
            if col == 1:
                ax.set_ylabel("")
            if row == 0:
                ax.set_title(
                    f"{window_s}秒非重叠窗口（n={len(data)}）",
                    fontsize=12,
                    fontweight="bold",
                    color=COLORS["dark"],
                    pad=7,
                )
            ax.text(
                0.02,
                0.94,
                method,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.1,
                fontweight="bold",
                color=METHOD_COLORS[method],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D8E0E6", alpha=0.9),
            )

    cbar_ax = fig.add_axes([0.953, 0.25, 0.014, 0.52])
    cbar = fig.colorbar(scatter_handle, cax=cbar_ax)
    cbar.set_label("平均速度（km/h）", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    notes = fig.add_subplot(outer[2])
    notes.axis("off")
    method_means = []
    for method in METHODS:
        data = all_results[(all_results["method"] == method) & (all_results["window_s"] == 60)]
        method_means.append(
            f"{method}：平均 k={data.density_veh_km.mean():.2f} veh/km，平均 q={data.flow_veh_h.mean():.1f} veh/h"
        )
    summary_left = (
        "数据与平均状态\n"
        + "\n".join(method_means)
        + f"\n范围内记录：{quality['selected_rows']:,}条；车辆：{quality['selected_vehicles']}辆；换道过渡段：{quality['lane_change_segments']}条"
    )
    summary_right = (
        "结果解释\n"
        "Edie轨迹裁剪与0.1秒空间采样的点位基本重合，说明位置和速度两条计算路径相互验证。\n"
        "虚拟检测器在10秒窗口中受整数车辆计数影响，点位呈离散条带；60秒聚合后与前两种方法明显接近。\n"
        "10秒窗口保留更多短时扰动，60秒窗口范围收缩、趋势更稳定；颜色统一表示窗口平均速度。"
    )
    notes.text(
        0.0,
        0.98,
        summary_left,
        va="top",
        fontsize=9.2,
        color=COLORS["dark"],
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F7F9", edgecolor="#D5DEE5"),
    )
    notes.text(
        0.53,
        0.98,
        summary_right,
        va="top",
        fontsize=9.2,
        color=COLORS["dark"],
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F5F0", edgecolor="#E4D9CB"),
    )

    fig.savefig(RESULT_DIR / "summary.png", dpi=230, facecolor="white")
    plt.close(fig)


def plot_visual_comparison(all_results: pd.DataFrame) -> None:
    """Compare A/B/C visual encodings for each of the three methods."""
    xlim, ylim = common_limits([all_results], pad=0.08)
    ten_all = all_results[all_results["window_s"] == 10]
    speed = ten_all["speed_km_h"].replace([np.inf, -np.inf], np.nan).dropna()
    speed_norm = Normalize(vmin=float(speed.min()), vmax=float(speed.max()))
    time_norm = Normalize(vmin=0.0, vmax=15.0)

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.0), sharex=True, sharey=True)
    speed_handle = None
    time_handle = None

    for row, method in enumerate(METHODS):
        ten = all_results[(all_results["method"] == method) & (all_results["window_s"] == 10)]
        sixty = all_results[(all_results["method"] == method) & (all_results["window_s"] == 60)]

        # A. Raw 10 s and 60 s scatter.
        ax = axes[row, 0]
        ax.scatter(
            ten.density_veh_km,
            ten.flow_veh_h,
            s=25,
            alpha=0.55,
            color=METHOD_COLORS[method],
            label="10秒",
        )
        ax.scatter(
            sixty.density_veh_km,
            sixty.flow_veh_h,
            s=58,
            alpha=0.9,
            facecolors="none",
            edgecolors=COLORS["dark"],
            linewidths=1.1,
            label="60秒",
        )
        if row == 0:
            ax.legend(frameon=False, fontsize=8.2, loc="upper right")

        # B. Speed-colored 10 s scatter.
        ax = axes[row, 1]
        speed_handle = ax.scatter(
            ten.density_veh_km,
            ten.flow_veh_h,
            c=ten.speed_km_h,
            cmap="viridis",
            norm=speed_norm,
            s=28,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.3,
        )

        # C. Chronological evolution for 10 s windows.
        ax = axes[row, 2]
        centers_min = (ten.start_s + ten.end_s) / 120.0
        ax.plot(
            ten.density_veh_km,
            ten.flow_veh_h,
            color="#8E99A4",
            lw=0.5,
            alpha=0.28,
            zorder=1,
        )
        time_handle = ax.scatter(
            ten.density_veh_km,
            ten.flow_veh_h,
            c=centers_min,
            cmap="plasma",
            norm=time_norm,
            s=28,
            alpha=0.84,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

        for col in range(3):
            ax = axes[row, col]
            style_axis(ax, xlim, ylim)
            ax.tick_params(labelsize=8.4)
            if row < 2:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")
            ax.text(
                0.02,
                0.95,
                method,
                transform=ax.transAxes,
                va="top",
                fontsize=8.8,
                fontweight="bold",
                color=METHOD_COLORS[method],
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D8E0E6", alpha=0.9),
            )

    column_titles = (
        "A  原始散点：10秒与60秒",
        "B  10秒窗口：按平均速度着色",
        "C  10秒窗口：按时间显示状态演化",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", color=COLORS["dark"], pad=9)

    speed_bar = fig.colorbar(speed_handle, cax=fig.add_axes([0.907, 0.16, 0.012, 0.68]))
    speed_bar.set_label("平均速度（km/h）", fontsize=9)
    speed_bar.ax.tick_params(labelsize=8)
    time_bar = fig.colorbar(time_handle, cax=fig.add_axes([0.948, 0.16, 0.012, 0.68]))
    time_bar.set_label("距16:00的分钟数", fontsize=9)
    time_bar.ax.tick_params(labelsize=8)

    fig.suptitle("三种计算方法下的流量—密度图表达对比", fontsize=17, fontweight="bold", color=COLORS["dark"], y=0.98)
    fig.text(
        0.5,
        0.947,
        "每一行对应一种计算口径；每一列只改变图形表达。所有面板使用相同坐标范围。",
        ha="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.065, right=0.89, top=0.91, bottom=0.07, hspace=0.26, wspace=0.18)
    fig.savefig(RESULT_DIR / "visualization_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def prepare_export_table(results: list[pd.DataFrame]) -> pd.DataFrame:
    keep = [
        "method",
        "window_s",
        "window_id",
        "start_s",
        "end_s",
        "density_veh_km",
        "flow_veh_h",
        "speed_km_h",
        "vehicle_count",
    ]
    return pd.concat([r[keep] for r in results], ignore_index=True).sort_values(
        ["method", "window_s", "window_id"]
    )


def build_quality_summary(df: pd.DataFrame, segments: pd.DataFrame, clipped: pd.DataFrame) -> dict:
    selected = df[
        (df["lane_id"] == LANE_ID)
        & (df["local_y_m"] >= Y_MIN)
        & (df["local_y_m"] < Y_MAX)
    ]
    near_area = (
        ((segments["y0"] >= Y_MIN) & (segments["y0"] < Y_MAX))
        | ((segments["y1"] >= Y_MIN) & (segments["y1"] < Y_MAX))
    )
    lane_changes = segments[
        near_area & (segments["lane0"] != segments["lane1"])
    ]
    return {
        "rows": len(df),
        "vehicles": int(df["vehicle_id"].nunique()),
        "duplicates": int(df.duplicated(["vehicle_id", "frame_id"]).sum()),
        "missing_critical": int(df[["vehicle_id", "time_s", "local_y_m", "lane_id"]].isna().sum().sum()),
        "selected_rows": len(selected),
        "selected_vehicles": int(selected["vehicle_id"].nunique()),
        "lane_change_segments": len(lane_changes),
        "clipped_segments": len(clipped),
        "selected_zero_speed_rows": int((selected["speed_m_s"] == 0).sum()),
    }


def write_summary(
    quality: dict,
    edie: dict[int, pd.DataFrame],
    snapshot: dict[int, pd.DataFrame],
    detector: dict[int, pd.DataFrame],
) -> None:
    e10, e60 = edie[10], edie[60]
    s10, s60 = snapshot[10], snapshot[60]
    d10, d60 = detector[10], detector[60]

    edie_10_to_60_k = e10.density_veh_km.to_numpy().reshape(-1, 6).mean(axis=1)
    edie_10_to_60_q = e10.flow_veh_h.to_numpy().reshape(-1, 6).mean(axis=1)
    consistency_k = float(np.max(np.abs(edie_10_to_60_k - e60.density_veh_km.to_numpy())))
    consistency_q = float(np.max(np.abs(edie_10_to_60_q - e60.flow_veh_h.to_numpy())))

    lines = [
        "NGSIM I-80 流量—密度基本图分析摘要",
        "=" * 44,
        "",
        "一、分析范围",
        "方向：I-80东向",
        f"车道：lane_id = {LANE_ID}",
        f"路段：local_y_m ∈ [{Y_MIN:.0f}, {Y_MAX:.0f}) m，长度 {LENGTH_M:.0f} m",
        f"虚拟检测器：local_y_m = {DETECTOR_Y:.0f} m",
        "时间：time_s ∈ [0, 900) s",
        "",
        "二、数据检查",
        f"总记录数：{quality['rows']:,}",
        f"总车辆数：{quality['vehicles']:,}",
        f"重复车辆-帧键：{quality['duplicates']}",
        f"关键字段缺失数：{quality['missing_critical']}",
        f"选定车道与路段记录数：{quality['selected_rows']:,}",
        f"选定车道与路段车辆数：{quality['selected_vehicles']}",
        f"选定范围零速度记录：{quality['selected_zero_speed_rows']}",
        f"范围附近换道过渡段：{quality['lane_change_segments']}",
        "",
        "三、Edie主结果",
        f"10秒窗口：n={len(e10)}，密度 {e10.density_veh_km.min():.2f}–{e10.density_veh_km.max():.2f} veh/km/lane，"
        f"流量 {e10.flow_veh_h.min():.2f}–{e10.flow_veh_h.max():.2f} veh/h/lane，"
        f"速度 {e10.speed_km_h.min():.2f}–{e10.speed_km_h.max():.2f} km/h",
        f"60秒窗口：n={len(e60)}，密度 {e60.density_veh_km.min():.2f}–{e60.density_veh_km.max():.2f} veh/km/lane，"
        f"流量 {e60.flow_veh_h.min():.2f}–{e60.flow_veh_h.max():.2f} veh/h/lane，"
        f"速度 {e60.speed_km_h.min():.2f}–{e60.speed_km_h.max():.2f} km/h",
        "",
        "四、方法一致性",
        f"Edie vs 0.1秒空间采样（10秒）密度MAE：{np.mean(np.abs(e10.density_veh_km-s10.density_veh_km)):.3f} veh/km",
        f"Edie vs 0.1秒空间采样（10秒）流量MAE：{np.mean(np.abs(e10.flow_veh_h-s10.flow_veh_h)):.3f} veh/h",
        f"Edie vs 虚拟检测器（10秒）密度MAE：{np.nanmean(np.abs(e10.density_veh_km-d10.density_veh_km)):.3f} veh/km",
        f"Edie vs 虚拟检测器（10秒）流量MAE：{np.nanmean(np.abs(e10.flow_veh_h-d10.flow_veh_h)):.3f} veh/h",
        f"Edie vs 0.1秒空间采样（60秒）密度MAE：{np.mean(np.abs(e60.density_veh_km-s60.density_veh_km)):.3f} veh/km",
        f"Edie vs 0.1秒空间采样（60秒）流量MAE：{np.mean(np.abs(e60.flow_veh_h-s60.flow_veh_h)):.3f} veh/h",
        f"Edie vs 虚拟检测器（60秒）密度MAE：{np.nanmean(np.abs(e60.density_veh_km-d60.density_veh_km)):.3f} veh/km",
        f"Edie vs 虚拟检测器（60秒）流量MAE：{np.nanmean(np.abs(e60.flow_veh_h-d60.flow_veh_h)):.3f} veh/h",
        f"Edie 6×10秒聚合到60秒最大密度差：{consistency_k:.10f} veh/km",
        f"Edie 6×10秒聚合到60秒最大流量差：{consistency_q:.10f} veh/h",
        "",
        "五、三种方法的15分钟平均状态",
        f"Edie轨迹裁剪：k={e60.density_veh_km.mean():.3f} veh/km，q={e60.flow_veh_h.mean():.3f} veh/h",
        f"0.1秒空间采样：k={s60.density_veh_km.mean():.3f} veh/km，q={s60.flow_veh_h.mean():.3f} veh/h",
        f"350米虚拟检测器：k={d60.density_veh_km.mean():.3f} veh/km，q={d60.flow_veh_h.mean():.3f} veh/h",
        "",
        "六、解释",
        "10秒窗口保留更多短时到达和启停波动，因此点云范围更广、离散度更高。",
        "60秒窗口对短时扰动进行平均，点云范围收缩，整体趋势更稳定。",
        "Edie法与空间采样法几乎重合；虚拟检测器在10秒窗口受整数车辆计数影响，60秒后明显接近。",
    ]
    (RESULT_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plotting()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    segments = build_segments(df)
    clipped = clip_segments_to_study_area(segments)
    quality = build_quality_summary(df, segments, clipped)

    edie = {w: calculate_edie(clipped, w) for w in WINDOWS}
    snapshot = {w: calculate_snapshot(df, w) for w in WINDOWS}
    detector = {w: calculate_detector(segments, w) for w in WINDOWS}

    results = [edie[10], edie[60], snapshot[10], snapshot[60], detector[10], detector[60]]
    export = prepare_export_table(results)
    export.to_csv(RESULT_DIR / "window_results.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    write_summary(quality, edie, snapshot, detector)
    plot_one_page(export, quality)
    plot_visual_comparison(export)

    print(f"完成。结果目录：{RESULT_DIR}")
    for path in sorted(RESULT_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()

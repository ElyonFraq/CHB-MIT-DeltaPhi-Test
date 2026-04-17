# -*- coding: utf-8 -*-
"""
CHB-MIT DeltaPhi pipeline (full GitHub-ready version)

This script keeps the clean GitHub-ready structure of the reduced version,
but reintroduces the broader analysis scope that existed in the original
Colab-heavy notebook/script:

- single-file DeltaPhi feature extraction
- optional multi-file combined analysis
- ROC-like threshold scans
- preictal-bin analysis
- late-preictal vs interictal analysis
- memory-integral analysis (J_memory)
- CSV/table/text export
- optional plot export

The script avoids Colab-only commands, Google Drive mounts, and hard-coded paths.
It is intended as a reproducible repo script, not as a notebook dump.

Example:
    python chb_mit_delta_phi_pipeline_full.py \
        --edf data/chb03_03.edf \
        --seizure-start 432 \
        --seizure-end 501 \
        --file-tag chb03_03 \
        --output-dir outputs

Multi-file example:
    python chb_mit_delta_phi_pipeline_full.py \
        --edf-list data/chb03_03.edf data/chb03_04.edf \
        --seizure-starts 432 2162 \
        --seizure-ends 501 2214 \
        --file-tags chb03_03 chb03_04 \
        --output-dir outputs \
        --export-plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

# Plotting is optional at runtime.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


ALPHA = 0.40
BETA = 0.35
GAMMA = 0.25

DEFAULT_WINDOW_SEC = 10
DEFAULT_SOP_SEC = 30 * 60
DEFAULT_SPH_SEC = 5 * 60
DEFAULT_POST_SEC = 30 * 60
DEFAULT_SMOOTH_WINDOWS = 5
DEFAULT_MEMORY_DECAY = 0.95

PREICTAL_BIN_EDGES = [5, 10, 15, 20, 25, 30]
PREICTAL_BIN_LABELS = ["5-10", "10-15", "15-20", "20-25", "25-30"]


def shannon_entropy_1d(x: np.ndarray, bins: int = 64) -> float:
    """Compute Shannon entropy for a 1D signal."""
    x = np.asarray(x).ravel()
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_window_features(window_data: np.ndarray) -> tuple[float, float, float]:
    """
    Compute simple S/I/C features for one EEG window.

    Parameters
    ----------
    window_data : np.ndarray
        Shape (n_channels, n_samples)

    Returns
    -------
    tuple[float, float, float]
        (S, I, C)
    """
    channel_vars = np.var(window_data, axis=1)
    channel_ranges = np.ptp(window_data, axis=1)
    s_val = 0.5 * np.mean(channel_vars) + 0.5 * np.mean(channel_ranges)

    entropies = [shannon_entropy_1d(ch, bins=64) for ch in window_data]
    i_val = float(np.mean(entropies))

    corr = np.corrcoef(window_data)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    c_val = float(np.mean(np.abs(upper)))

    return float(s_val), i_val, c_val


def build_windows(duration_sec: float, window_sec: int) -> pd.DataFrame:
    """Create consecutive non-overlapping time windows."""
    windows = []
    for start in np.arange(0, duration_sec, window_sec):
        end = min(start + window_sec, duration_sec)
        windows.append((float(start), float(end)))
    return pd.DataFrame(windows, columns=["win_start_s", "win_end_s"])


def add_labels(
    df_windows: pd.DataFrame,
    seizure_start: float,
    seizure_end: float,
    sop_sec: int,
    sph_sec: int,
    post_sec: int,
) -> pd.DataFrame:
    """Assign ictal/preictal/interictal_candidate/exclude labels."""

    def label_window(row: pd.Series) -> str:
        center = (row["win_start_s"] + row["win_end_s"]) / 2.0

        if seizure_start <= center < seizure_end:
            return "ictal"

        if (seizure_start - sop_sec) <= center < (seizure_start - sph_sec):
            return "preictal"

        if seizure_end <= center < (seizure_end + post_sec):
            return "exclude"

        return "interictal_candidate"

    df = df_windows.copy()
    df["label"] = df.apply(label_window, axis=1)
    return df


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_pipeline(
    edf_path: str,
    seizure_start: float,
    seizure_end: float,
    file_tag: str,
    output_dir: str = "outputs",
    window_sec: int = DEFAULT_WINDOW_SEC,
    sop_sec: int = DEFAULT_SOP_SEC,
    sph_sec: int = DEFAULT_SPH_SEC,
    post_sec: int = DEFAULT_POST_SEC,
    smooth_windows: int = DEFAULT_SMOOTH_WINDOWS,
) -> pd.DataFrame:
    """
    Run the DeltaPhi pipeline on one EDF recording.

    Returns
    -------
    pd.DataFrame
        Window-level feature table with ΔΦ values.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    sfreq = float(raw.info["sfreq"])
    duration_sec = raw.n_times / sfreq

    df_windows = build_windows(duration_sec, window_sec)
    df_windows = add_labels(
        df_windows=df_windows,
        seizure_start=seizure_start,
        seizure_end=seizure_end,
        sop_sec=sop_sec,
        sph_sec=sph_sec,
        post_sec=post_sec,
    )

    df_use = df_windows[
        df_windows["label"].isin(["preictal", "ictal", "interictal_candidate"])
    ].copy()

    rows: list[dict] = []
    for _, row in df_use.iterrows():
        start_sample = int(row["win_start_s"] * sfreq)
        end_sample = int(row["win_end_s"] * sfreq)
        segment = raw.get_data(start=start_sample, stop=end_sample)

        s_val, i_val, c_val = compute_window_features(segment)

        rows.append(
            {
                "file": file_tag,
                "edf_path": str(edf_path),
                "win_start_s": row["win_start_s"],
                "win_end_s": row["win_end_s"],
                "label": row["label"],
                "S": s_val,
                "I": i_val,
                "C": c_val,
            }
        )

    df_feat = pd.DataFrame(rows)
    if df_feat.empty:
        raise ValueError(f"No usable windows were produced from EDF: {edf_path}")

    baseline_df = df_feat[df_feat["label"] == "interictal_candidate"].copy()
    if baseline_df.empty:
        raise ValueError(
            f"No interictal_candidate windows found for {file_tag}. "
            "Baseline cannot be computed."
        )

    baseline_s = baseline_df["S"].mean()
    baseline_i = baseline_df["I"].mean()
    baseline_c = baseline_df["C"].mean()

    df_feat["dS"] = df_feat["S"] - baseline_s
    df_feat["dI"] = df_feat["I"] - baseline_i
    df_feat["dC"] = df_feat["C"] - baseline_c

    df_feat["DeltaPhi"] = (
        ALPHA * np.abs(df_feat["dS"])
        + BETA * np.abs(df_feat["dI"])
        + GAMMA * np.abs(df_feat["dC"])
    )

    df_feat["center_s"] = 0.5 * (df_feat["win_start_s"] + df_feat["win_end_s"])
    df_feat["DeltaPhi_smooth"] = (
        df_feat["DeltaPhi"].rolling(smooth_windows, min_periods=1).mean()
    )

    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / f"{file_tag}_features.csv"
    df_feat.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path}")
    print("\nLabel counts:")
    print(df_feat["label"].value_counts())
    print("\nDeltaPhi summary by label:")
    print(
        df_feat.groupby("label")["DeltaPhi"].agg(
            ["count", "mean", "std", "median", "max"]
        )
    )

    return df_feat


def summarize_by_label(df_feat: pd.DataFrame, value_col: str = "DeltaPhi") -> pd.DataFrame:
    """Summarize a metric by label."""
    return df_feat.groupby("label")[value_col].agg(
        ["count", "mean", "std", "median", "max"]
    )


def roc_like_threshold_scan(
    values: pd.Series,
    y_true: pd.Series,
    num_thresholds: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run a simple ROC-like threshold scan.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Full threshold table, and best row by (sensitivity + specificity).
    """
    if values.empty:
        raise ValueError("Cannot scan thresholds on empty value series.")

    thresholds = np.linspace(values.min(), values.max(), num_thresholds)
    results = []

    for t in thresholds:
        preds = (values > t).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)

        results.append(
            {
                "threshold": float(t),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "score": float(sensitivity + specificity),
            }
        )

    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results["score"].idxmax()].copy()
    return df_results, best


def evaluate_preictal_vs_interictal(
    df_feat: pd.DataFrame,
    value_col: str = "DeltaPhi_smooth",
    num_thresholds: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Evaluate preictal vs interictal_candidate using a ROC-like threshold scan."""
    df_eval = df_feat[df_feat["label"].isin(["preictal", "interictal_candidate"])].copy()
    if df_eval.empty:
        raise ValueError("No preictal/interictal_candidate rows found for evaluation.")
    df_eval["y"] = (df_eval["label"] == "preictal").astype(int)
    return roc_like_threshold_scan(df_eval[value_col], df_eval["y"], num_thresholds)


def add_minutes_before_seizure(
    df: pd.DataFrame,
    seizure_start: float,
    file_name: str,
) -> pd.DataFrame:
    """Annotate each row with minutes before seizure."""
    df = df.copy()
    df["file"] = file_name
    df["center_s"] = 0.5 * (df["win_start_s"] + df["win_end_s"])
    df["minutes_before_seizure"] = (seizure_start - df["center_s"]) / 60.0
    return df


def build_combined_preictal_table(
    dfs: Iterable[pd.DataFrame],
    seizure_starts: Iterable[float],
    file_tags: Iterable[str],
) -> pd.DataFrame:
    """Create combined table with minutes-before-seizure annotations."""
    parts = []
    for df, start, tag in zip(dfs, seizure_starts, file_tags):
        sub = add_minutes_before_seizure(df, start, tag)
        sub = sub[sub["label"].isin(["preictal", "interictal_candidate"])].copy()
        parts.append(sub)

    if not parts:
        raise ValueError("No dataframes provided for combined preictal table.")

    df_combined = pd.concat(parts, ignore_index=True)
    df_combined["preictal_bin"] = pd.cut(
        df_combined["minutes_before_seizure"],
        bins=PREICTAL_BIN_EDGES,
        labels=PREICTAL_BIN_LABELS,
        right=False,
    )
    return df_combined


def summarize_preictal_bins(
    df_pre_combined: pd.DataFrame,
    value_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize preictal bins and interictal means."""
    df_pre_only = df_pre_combined[df_pre_combined["label"] == "preictal"].copy()
    summary = df_pre_only.groupby("preictal_bin")[value_cols].agg(
        ["count", "mean", "std", "median", "max"]
    )
    interictal_means = {}
    for col in value_cols:
        interictal_means[col] = float(
            df_pre_combined[df_pre_combined["label"] == "interictal_candidate"][col].mean()
        )
    return summary, interictal_means


def evaluate_late_preictal_vs_interictal(
    df_pre_combined: pd.DataFrame,
    value_col: str,
    late_bin: str = "5-10",
    num_thresholds: int = 50,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Evaluate late preictal vs interictal_candidate."""
    df_late = df_pre_combined[
        (df_pre_combined["label"] == "interictal_candidate")
        | (
            (df_pre_combined["label"] == "preictal")
            & (df_pre_combined["preictal_bin"] == late_bin)
        )
    ].copy()

    if df_late.empty:
        raise ValueError("Late-preictal/interictal comparison produced no rows.")

    df_late["y"] = (
        (df_late["label"] == "preictal") & (df_late["preictal_bin"] == late_bin)
    ).astype(int)

    results, best = roc_like_threshold_scan(df_late[value_col], df_late["y"], num_thresholds)
    return results, best, df_late


def add_memory_integral(df: pd.DataFrame, decay: float = DEFAULT_MEMORY_DECAY) -> pd.DataFrame:
    """
    Add simple exponentially decaying cumulative memory term J_memory.

    J_t = decay * J_(t-1) + DeltaPhi_t
    """
    df = df.sort_values(["file", "center_s"]).copy()
    memory_values: list[tuple[int, float]] = []

    for _, sub in df.groupby("file"):
        j_val = 0.0
        for idx, row in sub.iterrows():
            j_val = decay * j_val + float(row["DeltaPhi"])
            memory_values.append((idx, j_val))

    memory_series = pd.Series(index=df.index, dtype=float)
    for idx, val in memory_values:
        memory_series.loc[idx] = val

    df["J_memory"] = memory_series
    return df.sort_index()


def export_plot_line(
    x: pd.Series,
    y: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    extra_hline: float | None = None,
) -> None:
    """Export a simple line plot if matplotlib is available."""
    if plt is None:
        print(f"Skipping plot (matplotlib unavailable): {out_path}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o")
    if extra_hline is not None:
        plt.axhline(extra_hline, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_plot_multi_series(
    series_map: dict[str, tuple[pd.Series, pd.Series]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Export multi-series line plot."""
    if plt is None:
        print(f"Skipping plot (matplotlib unavailable): {out_path}")
        return

    plt.figure(figsize=(14, 6))
    for label, (x, y) in series_map.items():
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_plot_roc(df_results: pd.DataFrame, out_path: Path, title: str) -> None:
    """Export ROC-like curve plot."""
    if plt is None:
        print(f"Skipping plot (matplotlib unavailable): {out_path}")
        return

    plt.figure(figsize=(6, 6))
    plt.plot(1 - df_results["specificity"], df_results["sensitivity"], marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_text_summary(summary_path: Path, sections: list[tuple[str, str]]) -> None:
    """Write human-readable summary text file."""
    with open(summary_path, "w", encoding="utf-8") as f:
        for title, body in sections:
            f.write(title + "\n")
            f.write("=" * len(title) + "\n")
            f.write(body.strip() + "\n\n")


def run_multi_analysis(
    edf_list: list[str],
    seizure_starts: list[float],
    seizure_ends: list[float],
    file_tags: list[str],
    output_dir: str,
    window_sec: int,
    sop_sec: int,
    sph_sec: int,
    post_sec: int,
    smooth_windows: int,
    memory_decay: float,
    export_plots: bool = False,
) -> dict[str, object]:
    """
    Run full multi-file analysis and export results.

    Returns
    -------
    dict[str, object]
        Collection of major analysis outputs.
    """
    if not (len(edf_list) == len(seizure_starts) == len(seizure_ends) == len(file_tags)):
        raise ValueError("edf_list, seizure_starts, seizure_ends, and file_tags must match in length.")

    out_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    tables_dir = ensure_dir(out_dir / "tables")

    feature_tables = []
    summaries = {}
    roc_tables = {}
    best_thresholds = {}

    for edf_path, start, end, tag in zip(edf_list, seizure_starts, seizure_ends, file_tags):
        df_feat = run_pipeline(
            edf_path=edf_path,
            seizure_start=start,
            seizure_end=end,
            file_tag=tag,
            output_dir=str(out_dir),
            window_sec=window_sec,
            sop_sec=sop_sec,
            sph_sec=sph_sec,
            post_sec=post_sec,
            smooth_windows=smooth_windows,
        )
        feature_tables.append(df_feat)

        summary = summarize_by_label(df_feat, "DeltaPhi")
        roc_df, best = evaluate_preictal_vs_interictal(df_feat, "DeltaPhi_smooth")

        summaries[tag] = summary
        roc_tables[tag] = roc_df
        best_thresholds[tag] = best

        summary.to_csv(tables_dir / f"{tag}_summary.csv")
        roc_df.to_csv(tables_dir / f"{tag}_roc.csv", index=False)

        if export_plots:
            export_plot_roc(roc_df, plots_dir / f"{tag}_roc_like.png", f"ROC-like curve ({tag})")

    df_combined = pd.concat(feature_tables, ignore_index=True)
    df_combined.to_csv(tables_dir / "combined_features.csv", index=False)

    summary_combined = summarize_by_label(df_combined, "DeltaPhi")
    roc_combined, best_combined = evaluate_preictal_vs_interictal(
        df_combined, "DeltaPhi_smooth", num_thresholds=60
    )

    summary_combined.to_csv(tables_dir / "summary_combined.csv")
    roc_combined.to_csv(tables_dir / "roc_combined.csv", index=False)

    # Preictal bin analysis (DeltaPhi / DeltaPhi_smooth)
    df_pre_combined = build_combined_preictal_table(
        feature_tables, seizure_starts, file_tags
    )
    df_pre_combined.to_csv(tables_dir / "combined_preictal_table.csv", index=False)

    summary_pre_bins, interictal_means = summarize_preictal_bins(
        df_pre_combined, ["DeltaPhi", "DeltaPhi_smooth"]
    )
    summary_pre_bins.to_csv(tables_dir / "summary_pre_bins.csv")

    late_results, best_late, df_late = evaluate_late_preictal_vs_interictal(
        df_pre_combined, "DeltaPhi_smooth", late_bin="5-10", num_thresholds=50
    )
    late_results.to_csv(tables_dir / "late_preictal_results.csv", index=False)
    df_late.to_csv(tables_dir / "late_preictal_rows.csv", index=False)

    # Memory analysis
    memory_tables = []
    for df_feat in feature_tables:
        memory_tables.append(add_memory_integral(df_feat, decay=memory_decay))

    df_mem_combined = pd.concat(memory_tables, ignore_index=True)
    df_mem_combined.to_csv(tables_dir / "memory_combined.csv", index=False)

    dfp = build_combined_preictal_table(memory_tables, seizure_starts, file_tags)
    summary_mem_bins, interictal_mem_means = summarize_preictal_bins(dfp, ["J_memory"])
    summary_mem_bins.to_csv(tables_dir / "summary_mem_bins.csv")

    mem_results, best_mem, df_late_mem = evaluate_late_preictal_vs_interictal(
        dfp, "J_memory", late_bin="5-10", num_thresholds=50
    )
    mem_results.to_csv(tables_dir / "memory_results.csv", index=False)
    df_late_mem.to_csv(tables_dir / "late_preictal_memory_rows.csv", index=False)

    # Optional plots
    if export_plots:
        export_plot_roc(roc_combined, plots_dir / "roc_combined.png", "ROC-like curve (combined)")
        export_plot_roc(late_results, plots_dir / "roc_late_preictal.png", "Late preictal (5-10) vs interictal")
        export_plot_roc(mem_results, plots_dir / "roc_memory_late_preictal.png", "Late preictal J_memory vs interictal")

        pre_bin_means = (
            df_pre_combined[df_pre_combined["label"] == "preictal"]
            .groupby("preictal_bin")["DeltaPhi"]
            .mean()
            .reindex(PREICTAL_BIN_LABELS)
        )
        export_plot_line(
            x=pd.Series(PREICTAL_BIN_LABELS),
            y=pre_bin_means.reset_index(drop=True),
            out_path=plots_dir / "preictal_bins_delta_phi.png",
            title="Preictal DeltaPhi by distance to seizure",
            xlabel="Minutes before seizure",
            ylabel="DeltaPhi",
            extra_hline=interictal_means["DeltaPhi"],
        )

        mem_bin_means = (
            dfp[dfp["label"] == "preictal"]
            .groupby("preictal_bin")["J_memory"]
            .mean()
            .reindex(PREICTAL_BIN_LABELS)
        )
        export_plot_line(
            x=pd.Series(PREICTAL_BIN_LABELS),
            y=mem_bin_means.reset_index(drop=True),
            out_path=plots_dir / "preictal_bins_j_memory.png",
            title="Preictal J_memory by distance to seizure",
            xlabel="Minutes before seizure",
            ylabel="J_memory",
            extra_hline=interictal_mem_means["J_memory"],
        )

        raw_series_map = {}
        smooth_series_map = {}
        memory_series_map = {}
        for tag, df in zip(file_tags, feature_tables):
            sub = df.sort_values("center_s")
            raw_series_map[f"{tag} raw"] = (sub["center_s"], sub["DeltaPhi"])
            smooth_series_map[f"{tag} smooth"] = (sub["center_s"], sub["DeltaPhi_smooth"])
        for tag, df in zip(file_tags, memory_tables):
            sub = df.sort_values("center_s")
            memory_series_map[f"{tag} memory"] = (sub["center_s"], sub["J_memory"])

        export_plot_multi_series(
            raw_series_map,
            plots_dir / "combined_raw_delta_phi.png",
            "DeltaPhi raw comparison across files",
            "Time (s)",
            "DeltaPhi",
        )
        export_plot_multi_series(
            smooth_series_map,
            plots_dir / "combined_smooth_delta_phi.png",
            "DeltaPhi smooth comparison across files",
            "Time (s)",
            "DeltaPhi smooth",
        )
        export_plot_multi_series(
            memory_series_map,
            plots_dir / "combined_memory_comparison.png",
            "J_memory comparison across files",
            "Time (s)",
            "J_memory",
        )

    summary_sections = []
    for tag in file_tags:
        summary_sections.append(
            (
                f"{tag}",
                "\n".join(
                    [
                        "DeltaPhi summary by label:",
                        summaries[tag].to_string(),
                        "",
                        "Best threshold (preictal vs interictal, using DeltaPhi_smooth):",
                        best_thresholds[tag].to_string(),
                    ]
                ),
            )
        )

    summary_sections.extend(
        [
            (
                "combined",
                "\n".join(
                    [
                        "DeltaPhi summary by label:",
                        summary_combined.to_string(),
                        "",
                        "Best combined threshold:",
                        best_combined.to_string(),
                    ]
                ),
            ),
            (
                "preictal_bins_delta_phi",
                "\n".join(
                    [
                        summary_pre_bins.to_string(),
                        "",
                        f"Interictal mean DeltaPhi: {interictal_means['DeltaPhi']}",
                        f"Interictal mean DeltaPhi_smooth: {interictal_means['DeltaPhi_smooth']}",
                        "",
                        "Best late-preictal threshold:",
                        best_late.to_string(),
                    ]
                ),
            ),
            (
                "memory_analysis",
                "\n".join(
                    [
                        summary_mem_bins.to_string(),
                        "",
                        f"Interictal mean J_memory: {interictal_mem_means['J_memory']}",
                        "",
                        "Best memory-based late-preictal threshold:",
                        best_mem.to_string(),
                    ]
                ),
            ),
        ]
    )
    write_text_summary(out_dir / "final_summary.txt", summary_sections)

    return {
        "feature_tables": feature_tables,
        "summaries": summaries,
        "roc_tables": roc_tables,
        "best_thresholds": best_thresholds,
        "df_combined": df_combined,
        "summary_combined": summary_combined,
        "roc_combined": roc_combined,
        "best_combined": best_combined,
        "df_pre_combined": df_pre_combined,
        "summary_pre_bins": summary_pre_bins,
        "late_results": late_results,
        "best_late": best_late,
        "df_mem_combined": df_mem_combined,
        "summary_mem_bins": summary_mem_bins,
        "mem_results": mem_results,
        "best_mem": best_mem,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the CHB-MIT DeltaPhi pipeline.")

    # Single-file mode
    parser.add_argument("--edf", help="Path to one EDF file")
    parser.add_argument("--seizure-start", type=float, help="Seizure start time in seconds")
    parser.add_argument("--seizure-end", type=float, help="Seizure end time in seconds")
    parser.add_argument("--file-tag", help="Short output tag, e.g. chb03_03")

    # Multi-file mode
    parser.add_argument("--edf-list", nargs="+", help="List of EDF files")
    parser.add_argument("--seizure-starts", nargs="+", type=float, help="List of seizure start times")
    parser.add_argument("--seizure-ends", nargs="+", type=float, help="List of seizure end times")
    parser.add_argument("--file-tags", nargs="+", help="List of file tags")

    # Shared config
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--window-sec", default=DEFAULT_WINDOW_SEC, type=int, help="Window size in seconds")
    parser.add_argument("--sop-sec", default=DEFAULT_SOP_SEC, type=int, help="Seizure occurrence period in seconds")
    parser.add_argument("--sph-sec", default=DEFAULT_SPH_SEC, type=int, help="Seizure prediction horizon in seconds")
    parser.add_argument("--post-sec", default=DEFAULT_POST_SEC, type=int, help="Postictal exclusion duration in seconds")
    parser.add_argument("--smooth-windows", default=DEFAULT_SMOOTH_WINDOWS, type=int, help="Rolling mean window count")
    parser.add_argument("--memory-decay", default=DEFAULT_MEMORY_DECAY, type=float, help="Decay factor for J_memory")
    parser.add_argument("--export-plots", action="store_true", help="Export PNG figures")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    # Multi-file mode
    if args.edf_list:
        if not (args.seizure_starts and args.seizure_ends and args.file_tags):
            raise ValueError(
                "Multi-file mode requires --edf-list, --seizure-starts, --seizure-ends, and --file-tags."
            )

        run_multi_analysis(
            edf_list=args.edf_list,
            seizure_starts=args.seizure_starts,
            seizure_ends=args.seizure_ends,
            file_tags=args.file_tags,
            output_dir=args.output_dir,
            window_sec=args.window_sec,
            sop_sec=args.sop_sec,
            sph_sec=args.sph_sec,
            post_sec=args.post_sec,
            smooth_windows=args.smooth_windows,
            memory_decay=args.memory_decay,
            export_plots=args.export_plots,
        )
        print("\nFull multi-file analysis completed.")
        return

    # Single-file mode
    if not (args.edf and args.seizure_start is not None and args.seizure_end is not None and args.file_tag):
        raise ValueError(
            "Single-file mode requires --edf, --seizure-start, --seizure-end, and --file-tag."
        )

    run_pipeline(
        edf_path=args.edf,
        seizure_start=args.seizure_start,
        seizure_end=args.seizure_end,
        file_tag=args.file_tag,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        sop_sec=args.sop_sec,
        sph_sec=args.sph_sec,
        post_sec=args.post_sec,
        smooth_windows=args.smooth_windows,
    )
    print("\nSingle-file pipeline completed.")


if __name__ == "__main__":
    main()

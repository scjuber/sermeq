"""
Day x cable-distance summary for the fin_energy_detector events in
fin_whale_detections_combined.csv -- a heatmap of detection counts by date and
representative channel (distance along the Morro Bay cable from the shore landing),
plus a per-day total bar underneath.

Usage:
    python plot_day_channel_summary.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})

REPO = Path("/home/sjuber/interferometry-for-orcas")
OUT = REPO / "results" / "localization_results" / "figures" / "33_day_channel_summary.png"

df = pd.read_csv(REPO / "results" / "fin_whale_detections_combined.csv")
det = df[df["source"] == "fin_energy_detector"].copy()
det["rep_channel_m"] = det["rep_channel_m"].astype(int)
det["date"] = det["date"].astype(str)

dates = sorted(det["date"].unique())
channels = sorted(det["rep_channel_m"].unique())
counts = (
    det.groupby(["date", "rep_channel_m"]).size()
    .reindex(pd.MultiIndex.from_product([dates, channels], names=["date", "rep_channel_m"]), fill_value=0)
    .unstack("rep_channel_m")
)

date_labels = [f"{d[4:6]}/{d[6:8]}" for d in dates]
daily_totals = counts.sum(axis=1)

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(12, 7), height_ratios=[4, 1.3], sharex=False
)

im = ax0.imshow(counts.values, aspect="auto", cmap="Blues", vmin=0)
ax0.set_xticks(range(len(channels)))
ax0.set_xticklabels([f"{c/1000:g}k" for c in channels])
ax0.set_yticks(range(len(dates)))
ax0.set_yticklabels(date_labels)
ax0.set_xlabel("Distance along cable from shore landing (m)")
ax0.set_ylabel("Date (2024)")
ax0.set_title("Fin whale detections by day and cable distance")
for i in range(len(dates)):
    for j in range(len(channels)):
        v = counts.values[i, j]
        if v > 0:
            ax0.text(j, i, int(v), ha="center", va="center",
                     color="white" if v > counts.values.max() * 0.55 else "black", fontsize=11)
fig.colorbar(im, ax=ax0, label="detections", pad=0.01, fraction=0.03)

ax1.bar(date_labels, daily_totals.values, color="#2a78d6")
for i, v in enumerate(daily_totals.values):
    ax1.text(i, v + 0.5, str(int(v)), ha="center", fontsize=12)
ax1.set_ylabel("detections / day")
ax1.set_ylim(0, daily_totals.max() * 1.2)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"total detections: {int(counts.values.sum())}  across {len(dates)} days, {len(channels)} channels")

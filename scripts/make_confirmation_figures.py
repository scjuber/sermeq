"""
Confirmation figures for localization, call type, and inter-call interval -- built from
already-saved CSVs (no DAS re-reads), so these are cheap and fast.

Produces:
  figures/12_call_type_centroid_histogram.png   -- corrected centroid freq, B/C split
  figures/13_ici_distributions.png              -- in-bout IPI + inter-bout gap histograms
  figures/14_localization_map.png               -- all localized events on the real cable route

Usage:
    python make_confirmation_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cable_geo import channel_to_lonlat

REPO = Path("/home/sjuber/interferometry-for-orcas")
FIGDIR = REPO / "results/localization_results/figures"
DATADIR = REPO / "results/localization_results/data"


def fig_call_type_histogram():
    notes = list(csv.DictReader(open(DATADIR / "call_type_notes.csv")))
    cf = np.array([float(r["centroid_freq_hz"]) for r in notes])
    types = np.array([r["note_type"] for r in notes])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(10, 45, 1.0)
    ax.hist(cf[types == "C"], bins=bins, color="#1b9e77", alpha=0.75, label=f"C -- classic (n={sum(types=='C')})")
    ax.hist(cf[types == "B"], bins=bins, color="#d95f02", alpha=0.85, label=f"B -- backbeat (n={sum(types=='B')})")
    ax.axvline(18.8, color="#d95f02", ls="--", lw=1.5, label="Archer et al. 2019: B centroid = 18.8 Hz")
    ax.axvline(23.7, color="#1b9e77", ls="--", lw=1.5, label="Archer et al. 2019: C centroid = 23.7 Hz")
    ax.axvline(21.25, color="k", ls=":", lw=1, label="classification threshold = 21.25 Hz")
    ax.set_xlabel("Strain-rate-corrected centroid frequency (Hz)")
    ax.set_ylabel("Notes")
    ax.set_title(f"Call-type classification: corrected centroid frequency, all {len(notes)} notes")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "12_call_type_centroid_histogram.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def fig_ici_distributions():
    rows = list(csv.DictReader(open(DATADIR / "calling_behavior_table.csv")))
    ipi = np.array([float(r["median_ipi_s"]) for r in rows if r["median_ipi_s"]])
    gaps = np.array([float(r["inter_bout_gap_s"]) / 60 for r in rows if r["inter_bout_gap_s"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(ipi, bins=np.arange(5, 30, 1.0), color="#7570b3", alpha=0.85)
    axes[0].axvline(np.median(ipi), color="k", ls="--", label=f"median = {np.median(ipi):.1f}s")
    axes[0].axvspan(7, 26, color="grey", alpha=0.08, label="detector's fin-song gate (7-26s)")
    axes[0].set_xlabel("Median in-bout inter-pulse interval (s)")
    axes[0].set_ylabel("Bouts"); axes[0].set_title(f"In-bout IPI (n={len(ipi)} bouts)")
    axes[0].legend(fontsize=9)

    axes[1].hist(gaps, bins=np.logspace(np.log10(max(gaps.min(), 0.5)), np.log10(gaps.max()), 25), color="#e7298a", alpha=0.85)
    axes[1].set_xscale("log")
    axes[1].axvline(np.median(gaps), color="k", ls="--", label=f"median = {np.median(gaps):.1f} min")
    axes[1].set_xlabel("Inter-bout gap, same day (min, log scale)")
    axes[1].set_ylabel("Bout pairs"); axes[1].set_title(f"Inter-bout gap (n={len(gaps)})")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "13_ici_distributions.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def fig_localization_map():
    rows = list(csv.DictReader(open(DATADIR / "event_localization.csv")))
    d_all = np.linspace(0, 10000, 4000)
    lon_route, lat_route = channel_to_lonlat(d_all)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(lon_route, lat_route, "-", color="grey", lw=1.5, zorder=1, label="cable route")

    loc = [r for r in rows if r["localized"] == "True"]
    unloc = [r for r in rows if r["localized"] != "True" and r["x_w"] not in ("", "nan")]
    for group, color, label, size, alpha in [
        (unloc, "#999999", f"detected, not localized (n={len(unloc)})", 18, 0.35),
        (loc, "#1c5f7c", f"localized (apex_visible + bursts) (n={len(loc)})", 34, 0.85),
    ]:
        xs = np.array([float(r["x_w"]) for r in group])
        lon, lat = channel_to_lonlat(xs)
        ax.scatter(lon, lat, s=size, color=color, alpha=alpha, label=label, zorder=2, edgecolor="none")

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"All {len(rows)} detected fin whale events, positioned along the real cable route")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_aspect(1 / np.cos(np.radians(35.35)))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGDIR / "14_localization_map.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_call_type_histogram()
    fig_ici_distributions()
    fig_localization_map()

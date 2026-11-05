"""
Localization diagnostic gallery: re-tracks arrival times and re-fits the apex model for
a diverse sample of events (good and bad, several days), plotting tracked picks + the
fitted moveout curve so the fit quality can be visually confirmed, not just trusted from
summary numbers. Includes 2 events that FAILED the localized gate, for honest contrast.

Usage:
    python make_localization_gallery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from batch_localize_events import (  # noqa: E402
    dphi_dir_for, files_around, load_padded_band_and_ref, matched_filter_track,
    nearest_file_for_epoch, validate_apex_fit, APERTURE_M, BEFORE_S, AFTER_S,
)
import csv

REPO = Path("/home/sjuber/interferometry-for-orcas")

EVENTS = [
    # (date, utc, rep_channel_m, expect_localized)
    ("20241106", "2024-11-06 23:24:14", 5400.0, True),
    ("20241108", "2024-11-08 19:53:24", 8400.0, True),
    ("20241112", "2024-11-12 15:28:24", 8400.0, True),
    ("20241110", "2024-11-10 21:05:04", 8100.0, True),
    ("20241107", "2024-11-07 12:10:04", 7200.0, True),
    ("20241109", "2024-11-09 18:08:24", 7500.0, True),
    ("20241107", "2024-11-07 17:38:24", 8400.0, True),
    ("20241110", "2024-11-10 09:36:44", 7500.0, True),
    ("20241106", "2024-11-06 15:35:54", 8400.0, True),
    ("20241106", "2024-11-06 04:54:14", 6600.0, False),   # apex_visible=True but bursts=False
    ("20241106", "2024-11-06 04:40:54", 8100.0, False),   # apex_visible=False
]


def get_seed(date, utc, rep_m):
    det_csv = REPO / f"results/fin_energy_MorroPhase_base_{date}/detections.csv"
    with det_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["utc"] == utc:
                peak0 = float(row["peak_times_s"].split("|")[0])
                return float(row["t_epoch"]) + peak0
    raise ValueError(f"event not found: {date} {utc}")


def run_event(date, utc, rep_m):
    seed_epoch = get_seed(date, utc, rep_m)
    center_time, ref_time_guess = nearest_file_for_epoch(date, seed_epoch)
    center_file = dphi_dir_for(date) / f"{center_time}.hdf5"
    dist_range_m = (max(0.0, rep_m - APERTURE_M), rep_m + APERTURE_M)
    pad_before, analysis_files, pad_after = files_around(date, center_time, BEFORE_S, AFTER_S)

    X, t_axis, x_positions, fs, dx, _ref = load_padded_band_and_ref(
        date, center_file, pad_before, analysis_files, pad_after, dist_range_m, rep_m)
    picked_t, picked_a, valid = matched_filter_track(X, t_axis, x_positions, dx, rep_m, ref_time_guess, fs)
    x, t = x_positions[valid], picked_t[valid]
    result = validate_apex_fit(x, t, t_axis, x_positions)
    return dict(x=x, t=t, fit=result["fit"], apex_visible=result["apex_visible"],
               bic_apex=result["bic_apex"], bic_straight=result["bic_straight"],
               x_positions=x_positions, t_axis=t_axis)


def main():
    n = len(EVENTS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.3 * nrows))
    axes = axes.flatten()

    for i, (date, utc, rep_m, expect) in enumerate(EVENTS):
        ax = axes[i]
        try:
            r = run_event(date, utc, rep_m)
            x, t, fit = r["x"], r["t"], r["fit"]
            good = r["bic_apex"] < r["bic_straight"]
            ax.scatter(x, t, s=6, alpha=0.5, color="#1c5f7c" if good else "#a8442f")
            xx = np.linspace(r["x_positions"].min(), r["x_positions"].max(), 300)
            tt = fit["t0"] + np.abs(xx - fit["x_w"]) / fit["v"]
            ax.plot(xx, tt, "r-", lw=1.5)
            ax.axvline(fit["x_w"], color="orange", ls=":", lw=1)
            status = "OK" if good else "BIC: straight-line preferred"
            ax.set_title(f"{date} {utc[11:]}\nx_w={fit['x_w']:.0f}m v={fit['v']:.0f}m/s "
                        f"rms={fit['rms_s']*1000:.0f}ms\n{status}", fontsize=9)
            print(f"{date} {utc}: x_w={fit['x_w']:.0f}m rms={fit['rms_s']*1000:.0f}ms "
                  f"apex_visible={r['apex_visible']} (expected localized={expect})", flush=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"error: {e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            print(f"{date} {utc}: ERROR {e}", flush=True)
        ax.set_xlabel("Distance along fiber (m)", fontsize=8)
        ax.set_ylabel("Arrival time (s)", fontsize=8)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Localization diagnostic gallery: tracked picks + fitted moveout, diverse sample "
                 "(blue/red = BIC favors apex; red curve = fit; last 2 panels are known-failed events, shown for contrast)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = REPO / "results/localization_results/figures/15_localization_diagnostic_gallery.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

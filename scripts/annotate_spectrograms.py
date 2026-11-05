"""
Regenerate a spectrogram+envelope panel for each exported detection WAV, overlaying
both the detector's own peak picks (lime, matches the original review-panel style)
and the user's Raven-confirmed call annotations (cyan spans = confirmed 20Hz,
orange = uncertain "HF?"), for direct visual comparison.

Uses the local WAV files in wav_detections/ (fast) instead of re-reading raw DAS
data from /1-fnp, since the WAV already contains the same representative-channel
trace the detector itself used.

Usage:
    python annotate_spectrograms.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, spectrogram

BAND_HZ = (10.0, 35.0)   # matches fin-whale-energy-detector.py's BAND_HZ default --
                          # the exported WAVs are broadband (detrended only), so the
                          # envelope/peak step must bandpass first to reproduce the
                          # original detector's own peak picks faithfully

REPO = Path("/home/sjuber/interferometry-for-orcas")
WAVDIR = REPO / "results" / "wav_detections"
OUTDIR = REPO / "results" / "wav_detections_annotated"
OUTDIR.mkdir(exist_ok=True)
SCRATCH = Path("/tmp/claude-39066/-home-sjuber-interferometry-for-orcas/52aa1143-2516-4749-b265-253b09d38c9c/scratchpad")

HEIGHT_MULT = 3.0
SMOOTH_S = 1.0
PEAK_MIN_SEP_S = 4.0

raven = json.load(open(SCRATCH / "_raven_mapped_spans.json"))
raven_by_file = defaultdict(list)
for r in raven:
    raven_by_file[r["file"]].append(r)

das = {row["png"].replace(".png", ".wav"): row
       for row in csv.DictReader(open(REPO / "results" / "fin_whale_detections_combined.csv"))
       if row["source"] == "fin_energy_detector"}


def annotate_one(fname):
    wav_path = WAVDIR / fname
    trace, fs = sf.read(str(wav_path))
    nt = len(trace)
    t_axis = np.arange(nt) / fs

    sos = butter(4, BAND_HZ, btype="band", fs=fs, output="sos")
    trace_band = sosfiltfilt(sos, trace)
    env = np.abs(hilbert(trace_band))
    smooth_n = max(3, int(round(SMOOTH_S * fs)) | 1)
    env_s = median_filter(env, size=smooth_n, mode="nearest")
    med = float(np.median(env_s))

    f_, ts_, Sxx = spectrogram(trace, fs=fs, nperseg=512, noverlap=int(0.9 * 512))
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    m = f_ <= min(60, fs / 2)

    rows = raven_by_file.get(fname, [])
    confirmed = [r for r in rows if r["label"] != "HF?"]
    uncertain = [r for r in rows if r["label"] == "HF?"]
    det_row = das.get(fname)
    fin_conc = det_row["fin_conc"] if det_row else "?"
    # Fresh local re-detection (matches render_event()'s own method exactly: bandpass
    # 10-35 Hz, Hilbert envelope, 1s median smoothing, 3x-median height, 4s min
    # separation), anchored at this clip's own t=0 -- NOT the CSV's peak_times_s
    # column, which is a confirmed bug upstream: event_row() reports t_epoch as
    # ev.t_start (earliest buffer in a merged multi-buffer event) but peak_times_s as
    # ev.best.peak_times_s (relative to whichever buffer scored highest, not
    # necessarily the first), so t_epoch + peak_times_s is wrong whenever those
    # differ. render_event() never hits this because it re-detects fresh from
    # ev.t_start, same as this function now does.
    peaks, _ = find_peaks(env_s, height=HEIGHT_MULT * med, distance=max(1, int(round(PEAK_MIN_SEP_S * fs))))
    peaks_s = [p / fs for p in peaks]

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    im = ax[0].pcolormesh(ts_, f_[m], Sxx_db[m], shading="auto", cmap="magma",
                           vmin=np.percentile(Sxx_db[m], 55), vmax=np.percentile(Sxx_db[m], 99.5))
    for p in peaks_s:
        ax[0].axvline(p, color="lime", lw=1.2, alpha=0.85)
    for r in confirmed:
        ax[0].axvspan(r["begin"], r["end"], color="cyan", alpha=0.25, lw=0)
        ax[0].axvline(r["begin"], color="cyan", lw=0.8, alpha=0.7)
    for r in uncertain:
        ax[0].axvspan(r["begin"], r["end"], color="orange", alpha=0.3, lw=0)
        ax[0].axvline(r["begin"], color="orange", lw=1.0, alpha=0.9)
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title(f"{fname}  fin_conc={fin_conc}\n"
                     f"lime=detector peaks ({len(peaks_s)})  cyan=confirmed 20Hz ({len(confirmed)})  "
                     f"orange=uncertain HF? ({len(uncertain)})")
    fig.colorbar(im, ax=ax[0], label="PSD (dB)", pad=0.01)

    ax[1].plot(t_axis, env_s, color="k", lw=0.8, label="smoothed envelope (10-35 Hz)")
    ax[1].axhline(HEIGHT_MULT * med, color="r", ls="--", lw=1, label=f"{HEIGHT_MULT:g}x median")
    for i, p in enumerate(peaks_s):
        ax[1].axvline(p, color="lime", lw=1.2, alpha=0.85, label="detector peaks" if i == 0 else None)
    for r in confirmed:
        ax[1].axvspan(r["begin"], r["end"], color="cyan", alpha=0.25, lw=0)
    for r in uncertain:
        ax[1].axvspan(r["begin"], r["end"], color="orange", alpha=0.3, lw=0)
    ax[1].set_ylabel("|Hilbert|")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = OUTDIR / fname.replace(".wav", "_annotated.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out.name, len(peaks_s), len(confirmed), len(uncertain)


def main():
    wavs = sorted(WAVDIR.glob("*.wav"))
    print(f"{len(wavs)} WAV files to annotate")
    for i, w in enumerate(wavs):
        try:
            out, npk, nconf, nunc = annotate_one(w.name)
            print(f"  [{i+1}/{len(wavs)}] {out}  det_peaks={npk} confirmed={nconf} uncertain={nunc}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{i+1}/{len(wavs)}] FAILED {w.name}: {e}", flush=True)
    print(f"\nwrote annotated panels to {OUTDIR}")


if __name__ == "__main__":
    main()

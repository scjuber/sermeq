"""
Energy detector for fin whale 20 Hz pulse trains in Morro Bay DAS.

Idea (matches the pseudocode in the task):
  for each rolling multi-file buffer, on a spread of fibre channels:
    X          = bandpass(strain-rate, 10-35 Hz)
    env        = |hilbert(X)|
    env_smooth = median_filter(env, ~1 s)
    peaks      = find_peaks(env_smooth, height = HEIGHT_MULT * median(env_smooth))
    if >= MIN_PEAKS peaks and several inter-peak gaps fall in the fin-song
       inter-pulse-interval band (IPI_MIN..IPI_MAX s):
        -> candidate detection

Consecutive / overlapping buffer hits are merged into "events".  For every event
the script writes a PNG review panel (spectrogram + smoothed envelope + t-x
waterfall) and one row in detections.csv, plus an index.html contact sheet so you
can scroll through and confirm (or reject) each one by eye.

Data layout (read straight off the HDF5 header, so 500 Hz base days and the
2 kHz 20241105 day both work):
  /1-fnp/petasaur/p-jbod1/morro/Morro_SiteData/<config>/<YYYYMMDD>/dphi/<HHMMSS>.hdf5
  data           (n_time, n_channel)  float32, strain-rate * header/dataScale
  header/dt      sample interval (s)      -> fs
  header/dx      channel spacing (m)
  header/time    UTC epoch of first sample

Examples
--------
  # one hour, quick look
  python fin-whale-energy-detector.py --config MorroPhase_base --date 20241108 \
      --start 20:00 --end 21:00

  # a whole day, 12 workers
  python fin-whale-energy-detector.py --config MorroPhase_base --date 20241108 --nproc 12

  # every base day
  for d in 20241106 20241107 20241108 20241109 20241110 20241111 20241112; do
      python fin-whale-energy-detector.py --config MorroPhase_base --date $d --nproc 12
  done
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import (butter, detrend, find_peaks, hilbert, sosfiltfilt,
                          spectrogram, welch)

# --------------------------------------------------------------------------- #
#  detector configuration (CLI can override the ones that matter)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/1-fnp/petasaur/p-jbod1/morro/Morro_SiteData")

BAND_HZ = (10.0, 35.0)          # fin 20 Hz pulse energy lives ~15-30 Hz
SMOOTH_S = 1.0                  # median-filter window on the envelope
HEIGHT_MULT = 3.0              # peak height threshold = HEIGHT_MULT * median(env_smooth)
MIN_PEAKS = 3                  # need at least this many peaks in the buffer
PEAK_MIN_SEP_S = 4.0           # fin 20 Hz pulses are not closer than ~5 s
IPI_MIN_S, IPI_MAX_S = 7.0, 26.0   # classic fin-song inter-pulse interval band
MIN_GOOD_IPIS = 2             # need >=2 inter-peak gaps inside that band

FIN_PULSE_HZ = (16.0, 26.0)   # where the 20 Hz pulse energy concentrates
MIN_FIN_CONC = 0.0            # fraction of CONC_REF_HZ energy that must sit in FIN_PULSE_HZ
                              # (0 = disabled by default; superseded by MIN_FIN_CONC_NORM below,
                              # which is what actually distinguishes a tonal from broadband noise)
MAX_IPI_CV = None             # optional: cap coeff. of variation of in-band IPIs (regularity)

# broadband-rejection gate: the OLD fin_conc (pulse-band energy / search-band energy)
# is a near-useless statistic when pulse_band spans most of band's width -- e.g.
# band=(25,55), pulse_band=(30,50) means FLAT WHITE NOISE already scores (50-30)/(55-25)
# = 0.667 by construction, no tonal structure required. That's exactly what let two
# broadband/mechanical pulses through a min_fin_conc=0.30 gate on the 40 Hz pilot scan.
# Fix: measure concentration against a genuinely wide reference spectrum (independent of
# the search band), then normalize by the pulse/reference bandwidth ratio so the score
# means "how many times denser is the energy in pulse_band than flat noise would give" --
# a real peakedness/tonality measure, not a tautological overlap fraction.
CONC_REF_HZ = (2.0, 150.0)    # wide reference band for the normalized concentration gate
MIN_FIN_CONC_NORM = 3.0       # pulse-band energy density must be >= this x the reference-band
                              # average density (1.0 = indistinguishable from flat/white noise)

BUFFER_FILES = 12             # rolling window length (files); 12 * 10 s = 120 s
BUFFER_OVERLAP_FILES = 2      # step = BUFFER_FILES - overlap

# channels to listen at: DIST_MIN..DIST_MAX m, every DIST_STEP m, stacked +/- STACK_HALF_M
# the first ~1.5 km is surf-zone / landing noise; whale calls in the site notebooks
# show up 6-8.5 km out, but keep the search wide and let the fin-conc gate cull junk
DIST_MIN_M, DIST_MAX_M, DIST_STEP_M = 1500.0, 8600.0, 300.0
STACK_HALF_M = 12.0

# review-panel rendering
PANEL_MAX_SPAN_S = 200.0      # cap how much of a long bout we render
WATERFALL_HALF_M = 300.0      # +/- distance shown in the t-x panel


# --------------------------------------------------------------------------- #
#  file discovery / IO
# --------------------------------------------------------------------------- #
def list_day_files(config: str, date: str) -> list[Path]:
    d = DATA_ROOT / config / date / "dphi"
    if not d.is_dir():
        raise FileNotFoundError(f"no such directory: {d}")
    return sorted(d.glob("*.hdf5"))


def file_epoch(path: Path) -> float | None:
    """None on an unreadable file (e.g. a truncated write) -- caller filters these out."""
    try:
        with h5py.File(path, "r", locking=False) as fp:
            return float(fp["header/time"][()])
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! unreadable file {path.name}: {e}")
        return None


def read_meta(path: Path) -> dict:
    with h5py.File(path, "r", locking=False) as fp:
        dt = float(fp["header/dt"][()])
        return dict(
            fs=1.0 / dt,
            dx=float(fp["header/dx"][()]),
            scale=float(fp["header/dataScale"][()]),
            n_time=int(fp["data"].shape[0]),
            n_chan=int(fp["data"].shape[1]),
            t0=float(fp["header/time"][()]),
        )


def read_block(files: list[Path], ch_lo: int, ch_hi: int, scale: float) -> np.ndarray:
    """Vertically stack data[:, ch_lo:ch_hi] over `files` -> (n_time_total, ch_hi-ch_lo)."""
    blocks = []
    for f in files:
        with h5py.File(f, "r", locking=False) as fp:
            blocks.append(np.asarray(fp["data"][:, ch_lo:ch_hi], dtype=np.float32))
    return np.vstack(blocks) * scale


def read_span(files: list[Path], cols: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    """Read the *contiguous* column span cols.min()..cols.max() and stack over files.

    h5py fancy-indexes a non-chunked dataset one hyperslab per column -- reading 550
    scattered columns costs ~20x a contiguous read of the whole span between them
    (measured on petasaur). So we read the span and let the caller slice in memory.
    Returns (block, ch_lo) where block[:, c - ch_lo] is fibre column c.
    """
    ch_lo, ch_hi = int(cols.min()), int(cols.max()) + 1
    return read_block(files, ch_lo, ch_hi, scale), ch_lo


# --------------------------------------------------------------------------- #
#  core energy detector (runs on one rolling buffer)
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    t_start: float                  # UTC epoch of first sample in the buffer
    buf_span_s: float
    dist_m: float                   # representative channel distance
    peak_times_s: list[float]       # peak offsets from t_start (s)
    ipis_s: list[float]
    good_ipis_s: list[float]
    median_ipi_s: float
    ipi_cv: float                   # std/mean of in-band IPIs (0 = perfectly regular)
    fin_conc: float                 # frac. of CONC_REF_HZ energy inside FIN_PULSE_HZ
    fin_conc_norm: float            # fin_conc / (pulse_bw/ref_bw) -- 1.0 = flat/white noise,
                                    # >>1 = real tonal peak; this is the broadband-rejection score
    snr: float                      # max(env_smooth) / median(env_smooth)
    n_peaks: int
    files: list[str] = field(default_factory=list)


def detect_in_buffer(
    files: list[Path],
    meta: dict,
    channel_dists: np.ndarray,
    *,
    band=BAND_HZ,
    smooth_s=SMOOTH_S,
    height_mult=HEIGHT_MULT,
    min_peaks=MIN_PEAKS,
    ipi_min=IPI_MIN_S,
    ipi_max=IPI_MAX_S,
    min_good_ipis=MIN_GOOD_IPIS,
    min_fin_conc=MIN_FIN_CONC,
    pulse_band=FIN_PULSE_HZ,
    max_ipi_cv=MAX_IPI_CV,
    conc_ref_hz=CONC_REF_HZ,
    min_fin_conc_norm=MIN_FIN_CONC_NORM,
) -> list[Candidate]:
    fs, dx, scale = meta["fs"], meta["dx"], meta["scale"]
    n_chan = meta["n_chan"]

    stack_half = max(1, int(round(STACK_HALF_M / dx)))
    centres = np.array([int(round(d / dx)) for d in channel_dists])
    centres = centres[(centres - stack_half >= 0) & (centres + stack_half + 1 <= n_chan)]
    if centres.size == 0:
        return []

    # Read the contiguous column span in one shot (fast IO on this non-chunked
    # dataset), collapse each channel's +/- stack strip to a single trace, and only
    # THEN detrend/filter -- carrying all ~7000 raw columns through sosfiltfilt is
    # what makes this CPU-bound, and the stack-mean commutes with the linear filter.
    strips = [np.arange(c - stack_half, c + stack_half + 1) for c in centres]
    cols = np.unique(np.concatenate(strips))
    span, ch_lo = read_span(files, cols, scale)
    traces = np.stack([span[:, strip - ch_lo].mean(axis=1) for strip in strips], axis=1)
    del span                                                  # (nt, n_centres)

    raw = detrend(traces, axis=0)
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    block = sosfiltfilt(sos, raw, axis=0)

    nt = block.shape[0]
    buf_span_s = nt / fs
    t_start = meta["t0"]
    smooth_n = max(3, int(round(smooth_s * fs)) | 1)       # odd
    min_sep_n = max(1, int(round(PEAK_MIN_SEP_S * fs)))

    out: list[Candidate] = []
    for j, c in enumerate(centres):
        trace = block[:, j]
        env = np.abs(hilbert(trace))
        env_s = median_filter(env, size=smooth_n, mode="nearest")

        med = float(np.median(env_s))
        if med <= 0:
            continue
        peaks, _ = find_peaks(env_s, height=height_mult * med, distance=min_sep_n)
        if peaks.size < min_peaks:
            continue

        ipis = np.diff(peaks) / fs
        good = ipis[(ipis >= ipi_min) & (ipis <= ipi_max)]
        if good.size < min_good_ipis:
            continue

        ipi_cv = float(np.std(good) / np.mean(good)) if good.size >= 2 else np.nan
        if max_ipi_cv is not None and not (ipi_cv <= max_ipi_cv):
            continue

        # is the pulse-train energy really concentrated in the target pulse band, or
        # just broadband junk? Reference band is now CONC_REF_HZ, wide and independent
        # of the search `band` -- using `band` itself as the reference (old behavior)
        # made the ratio tautological whenever pulse_band covers most of band's width
        # (e.g. band=(25,55), pulse_band=(30,50): FLAT WHITE NOISE scores 0.667 there by
        # construction, no tonal structure needed -- exactly how 2 broadband/mechanical
        # pulses slipped through a min_fin_conc=0.30 gate on the 40 Hz pilot scan).
        # fin_conc_norm divides out that bandwidth-ratio bias: it's energy density in
        # pulse_band relative to the average density across the wide reference band, so
        # 1.0 = indistinguishable from flat noise, and only a real spectral peak scores
        # meaningfully above that.
        ref_lo = max(conc_ref_hz[0], 0.5)
        ref_hi = min(conc_ref_hz[1], fs / 2 - 1.0)
        fpx, pxx = welch(raw[:, j], fs=fs, nperseg=min(1024, nt))
        wide = (fpx >= ref_lo) & (fpx <= ref_hi)
        finb = (fpx >= pulse_band[0]) & (fpx <= pulse_band[1])
        wtot = float(np.trapezoid(pxx[wide], fpx[wide]))
        fin_conc = float(np.trapezoid(pxx[finb], fpx[finb]) / wtot) if wtot > 0 else 0.0
        pulse_bw = pulse_band[1] - pulse_band[0]
        ref_bw = ref_hi - ref_lo
        null_frac = pulse_bw / ref_bw if ref_bw > 0 else 1.0
        fin_conc_norm = fin_conc / null_frac if null_frac > 0 else 0.0
        if fin_conc < min_fin_conc:
            continue
        if fin_conc_norm < min_fin_conc_norm:
            continue

        out.append(
            Candidate(
                t_start=t_start,
                buf_span_s=buf_span_s,
                dist_m=float(c * dx),
                peak_times_s=[float(p / fs) for p in peaks],
                ipis_s=[round(float(v), 2) for v in ipis],
                good_ipis_s=[round(float(v), 2) for v in good],
                ipi_cv=round(ipi_cv, 3),
                fin_conc=round(fin_conc, 3),
                fin_conc_norm=round(fin_conc_norm, 3),
                median_ipi_s=round(float(np.median(good)), 2),
                snr=round(float(env_s.max() / med), 2),
                n_peaks=int(peaks.size),
                files=[f.name for f in files],
            )
        )
    return out


def _worker(idx: int, *, all_files, epochs, meta, channel_dists, det_kwargs):
    lo = idx
    hi = min(idx + BUFFER_FILES, len(all_files))
    files = all_files[lo:hi]
    if len(files) < BUFFER_FILES // 2:
        return []
    # skip buffers with a time gap (missing files) larger than 2 nominal file lengths
    span = epochs[hi - 1] - epochs[lo]
    nominal = (hi - 1 - lo) * (meta["n_time"] / meta["fs"])
    if span > nominal + 2 * (meta["n_time"] / meta["fs"]):
        return []
    m = dict(meta, t0=epochs[lo])
    try:
        return detect_in_buffer(files, m, channel_dists, **det_kwargs)
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! buffer {files[0].name}: {e}")
        return []


# --------------------------------------------------------------------------- #
#  merge candidates -> events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    t_start: float
    t_end: float
    best: Candidate
    channels_m: list[float]
    n_candidates: int

    @property
    def utc(self) -> datetime:
        return datetime.fromtimestamp(self.t_start, tz=timezone.utc)


def merge_candidates(cands: list[Candidate]) -> list[Event]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: (c.t_start, c.dist_m))
    events: list[Event] = []
    cur: list[Candidate] = [cands[0]]

    def span(c: Candidate) -> tuple[float, float]:
        return c.t_start, c.t_start + c.buf_span_s

    for c in cands[1:]:
        c_lo, _ = span(c)
        cur_hi = max(span(x)[1] for x in cur)
        if c_lo <= cur_hi:            # overlaps the running bout
            cur.append(c)
        else:
            events.append(_finish_event(cur))
            cur = [c]
    events.append(_finish_event(cur))
    return events


def _finish_event(group: list[Candidate]) -> Event:
    best = max(group, key=lambda c: (len(c.good_ipis_s), c.fin_conc_norm, c.snr))
    return Event(
        t_start=min(c.t_start for c in group),
        t_end=max(c.t_start + c.buf_span_s for c in group),
        best=best,
        channels_m=sorted({round(c.dist_m) for c in group}),
        n_candidates=len(group),
    )


# --------------------------------------------------------------------------- #
#  review panels
# --------------------------------------------------------------------------- #
def render_event(ev: Event, config: str, date: str, all_files, epochs, meta, outdir: Path,
                 band=BAND_HZ) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs, dx, scale, n_chan = meta["fs"], meta["dx"], meta["scale"], meta["n_chan"]

    span_s = min(PANEL_MAX_SPAN_S, ev.t_end - ev.t_start)
    lo = int(np.searchsorted(epochs, ev.t_start - 1e-3))
    hi = int(np.searchsorted(epochs, ev.t_start + span_s + 1e-3))
    hi = min(max(hi, lo + 1), len(all_files))
    files = all_files[lo:hi]
    t0 = epochs[lo]

    c_rep = int(round(ev.best.dist_m / dx))
    wf_half = int(round(WATERFALL_HALF_M / dx))
    stack_half = max(1, int(round(STACK_HALF_M / dx)))
    ch_lo = max(0, c_rep - wf_half)
    ch_hi = min(n_chan, c_rep + wf_half + 1)

    block = read_block(files, ch_lo, ch_hi, scale)
    block = detrend(block, axis=0)
    # use the SAME band the detector actually ran with -- previously hardcoded to the
    # module-level 20 Hz-call default, so any --band override (e.g. a 40 Hz search)
    # produced panels bandpassed to the wrong band and showing no usable signal.
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    block = sosfiltfilt(sos, block, axis=0)
    nt = block.shape[0]
    t_axis = np.arange(nt) / fs

    ci = c_rep - ch_lo
    trace = block[:, max(0, ci - stack_half): ci + stack_half + 1].mean(axis=1)
    env = np.abs(hilbert(trace))
    smooth_n = max(3, int(round(SMOOTH_S * fs)) | 1)
    env_s = median_filter(env, size=smooth_n, mode="nearest")
    med = float(np.median(env_s))
    peaks, _ = find_peaks(env_s, height=HEIGHT_MULT * med,
                          distance=max(1, int(round(PEAK_MIN_SEP_S * fs))))

    f_, ts_, Sxx = spectrogram(trace, fs=fs, nperseg=512,
                               noverlap=int(0.9 * 512))
    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    m = f_ <= 60
    im = ax[0].pcolormesh(ts_, f_[m], Sxx_db[m], shading="auto",
                          vmin=np.percentile(Sxx_db[m], 55),
                          vmax=np.percentile(Sxx_db[m], 99.5), cmap="magma")
    ax[0].axhspan(band[0], band[1], color="cyan", alpha=0.12)
    for p in peaks:
        ax[0].axvline(p / fs, color="lime", lw=0.8, alpha=0.7)
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title(
        f"{config} {date}  {ev.utc:%H:%M:%S} UTC   rep. channel {ev.best.dist_m:.0f} m\n"
        f"n_peaks={ev.best.n_peaks}  IPIs={ev.best.ipis_s}  "
        f"in-band={ev.best.good_ipis_s}  median={ev.best.median_ipi_s}s  "
        f"IPI cv={ev.best.ipi_cv}  fin-conc={ev.best.fin_conc}  "
        f"fin-conc-norm={ev.best.fin_conc_norm}  SNR={ev.best.snr}"
    )
    fig.colorbar(im, ax=ax[0], label="PSD (dB)", pad=0.01)

    ax[1].plot(t_axis, env_s, color="k", lw=0.8, label="smoothed envelope")
    ax[1].axhline(HEIGHT_MULT * med, color="r", ls="--", lw=1,
                  label=f"{HEIGHT_MULT:g}x median")
    ax[1].plot(peaks / fs, env_s[peaks], "v", color="lime", ms=9, label="peaks")
    ax[1].set_ylabel("|hilbert| (a.u.)")
    ax[1].legend(loc="upper right", fontsize=9)

    dist_axis = (ch_lo + np.arange(block.shape[1])) * dx
    vmax = np.percentile(np.abs(block), 99)
    ax[2].pcolormesh(t_axis, dist_axis, block.T, shading="auto",
                     vmin=-vmax, vmax=vmax, cmap="seismic")
    ax[2].axhline(ev.best.dist_m, color="lime", ls=":", lw=1)
    for p in peaks:
        ax[2].axvline(p / fs, color="k", lw=0.5, alpha=0.4)
    ax[2].set_ylabel("Distance along fibre (m)")
    ax[2].set_xlabel(f"Time (s) since {ev.utc:%Y-%m-%d %H:%M:%S} UTC")

    fig.tight_layout()
    fname = f"{config}_{date}_{ev.utc:%H%M%S}_ch{ev.best.dist_m:.0f}m.png"
    fig.savefig(outdir / fname, dpi=110)
    plt.close(fig)
    return fname


def event_row(ev: Event, config: str, date: str, png: str = "") -> dict:
    # peak_times_s must be relative to t_epoch (ev.t_start), but ev.best may be a
    # later-starting candidate within the merged group -- shift by the offset between
    # the winning candidate's own t_start and the group's t_start to correct for that.
    peak_offset = ev.best.t_start - ev.t_start
    return dict(
        config=config, date=date,
        utc=f"{ev.utc:%Y-%m-%d %H:%M:%S}",
        t_epoch=f"{ev.t_start:.3f}",
        rep_channel_m=f"{ev.best.dist_m:.0f}",
        n_peaks=ev.best.n_peaks,
        ipis_s="|".join(f"{v:g}" for v in ev.best.ipis_s),
        good_ipis_s="|".join(f"{v:g}" for v in ev.best.good_ipis_s),
        median_ipi_s=ev.best.median_ipi_s,
        ipi_cv=ev.best.ipi_cv,
        fin_conc=ev.best.fin_conc,
        fin_conc_norm=ev.best.fin_conc_norm,
        snr=ev.best.snr,
        n_candidates=ev.n_candidates,
        channels_m="|".join(str(c) for c in ev.channels_m),
        peak_times_s="|".join(f"{v + peak_offset:g}" for v in ev.best.peak_times_s),
        png=png,
    )


def write_csv(rows: list[dict], csv_path: Path) -> None:
    cols = (list(rows[0].keys()) if rows else
            ["config", "date", "utc", "t_epoch", "rep_channel_m", "n_peaks",
             "ipis_s", "good_ipis_s", "median_ipi_s", "ipi_cv", "fin_conc",
             "fin_conc_norm", "snr", "n_candidates", "channels_m", "peak_times_s", "png"])
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _render_worker(args) -> tuple[int, str]:
    k, ev, config, date, all_files, epochs, meta, outdir, band = args
    try:
        return k, render_event(ev, config, date, all_files, epochs, meta, outdir, band=band)
    except Exception as exc:                                 # noqa: BLE001
        print(f"  ! render {ev.utc:%H:%M:%S}: {exc}")
        return k, ""


def write_index(events: list[Event], rows: list[dict], config: str, date: str, outdir: Path) -> None:
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>fin-whale energy detections {config} {date}</title>",
        "<style>body{font:14px system-ui;margin:24px;background:#fafafa}"
        "figure{margin:0 0 32px;border:1px solid #ddd;background:#fff;padding:8px}"
        "img{width:100%;height:auto}"
        "figcaption{font-family:monospace;white-space:pre-wrap;padding:6px 2px}</style>",
        f"<h1>{config} {date} &mdash; {len(events)} fin-whale energy events</h1>",
        f"<p>band {BAND_HZ[0]:g}-{BAND_HZ[1]:g} Hz &middot; "
        f"height {HEIGHT_MULT:g}&times;median &middot; "
        f"IPI {IPI_MIN_S:g}-{IPI_MAX_S:g} s &middot; &ge;{MIN_PEAKS} peaks &middot; "
        f"&ge;{MIN_GOOD_IPIS} in-band gaps. Confirm each by eye.</p>",
    ]
    for ev, row in zip(events, rows):
        parts.append(
            f"<figure><img src='{row['png']}' loading='lazy'>"
            f"<figcaption>{ev.utc:%Y-%m-%d %H:%M:%S} UTC   "
            f"channel {row['rep_channel_m']} m   n_peaks {row['n_peaks']}   "
            f"IPIs {row['ipis_s']}   in-band {row['good_ipis_s']}   "
            f"median IPI {row['median_ipi_s']} s   IPI cv {row['ipi_cv']}   "
            f"fin-conc {row['fin_conc']}   fin-conc-norm {row['fin_conc_norm']}   SNR {row['snr']}   "
            f"buffers {row['n_candidates']}   channels_hit {row['channels_m']}</figcaption></figure>"
        )
    (outdir / "index.html").write_text("\n".join(parts))


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def parse_hhmm(s: str | None) -> float | None:
    if s is None:
        return None
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def main() -> None:
    global BUFFER_FILES, BUFFER_OVERLAP_FILES
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="MorroPhase_base")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--start", help="HH:MM UTC (restrict scan)")
    ap.add_argument("--end", help="HH:MM UTC (restrict scan)")
    ap.add_argument("--dist-min", type=float, default=DIST_MIN_M)
    ap.add_argument("--dist-max", type=float, default=DIST_MAX_M)
    ap.add_argument("--dist-step", type=float, default=DIST_STEP_M)
    ap.add_argument("--band", type=float, nargs=2, default=list(BAND_HZ))
    ap.add_argument("--height-mult", type=float, default=HEIGHT_MULT)
    ap.add_argument("--min-peaks", type=int, default=MIN_PEAKS)
    ap.add_argument("--ipi", type=float, nargs=2, default=[IPI_MIN_S, IPI_MAX_S])
    ap.add_argument("--min-good-ipis", type=int, default=MIN_GOOD_IPIS)
    ap.add_argument("--min-fin-conc", type=float, default=MIN_FIN_CONC,
                    help="min frac of --conc-ref energy in --pulse-band (0 disables; "
                         "legacy gate, superseded by --min-fin-conc-norm)")
    ap.add_argument("--pulse-band", type=float, nargs=2, default=list(FIN_PULSE_HZ),
                    help="core band the pulse energy must concentrate in, e.g. "
                         "16 26 for the classic 20 Hz pulse, 30 50 for a 40 Hz call")
    ap.add_argument("--conc-ref", type=float, nargs=2, default=list(CONC_REF_HZ),
                    help="wide reference band for the normalized concentration gate, "
                         "independent of --band/--pulse-band")
    ap.add_argument("--min-fin-conc-norm", type=float, default=MIN_FIN_CONC_NORM,
                    help="min pulse-band energy density relative to --conc-ref average "
                         "density (1.0 = flat/white noise, higher = more tonal/peaked; "
                         "this is the broadband-rejection gate, 0 disables)")
    ap.add_argument("--max-ipi-cv", type=float, default=None,
                    help="cap coeff. of variation of in-band IPIs (regularity gate)")
    ap.add_argument("--buffer-files", type=int, default=BUFFER_FILES,
                    help="rolling window length in files (10s/file); the 20 Hz default "
                         "(12 -> 120s) is too short to catch >=2 pulses of a call type "
                         "with a longer/more irregular IPI (e.g. 40 Hz calls, ICI ~40-60s)")
    ap.add_argument("--buffer-overlap-files", type=int, default=BUFFER_OVERLAP_FILES)
    ap.add_argument("--nproc", type=int, default=6,
                    help="worker processes (petasaur IO saturates fast; 4-8 is plenty)")
    ap.add_argument("--max-events", type=int, default=400, help="cap panels rendered")
    ap.add_argument("--scan-only", "--no-plots", dest="scan_only", action="store_true",
                    help="detect + write detections.csv, skip the PNG panels")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    BUFFER_FILES = args.buffer_files
    BUFFER_OVERLAP_FILES = args.buffer_overlap_files

    outdir = Path(args.outdir or f"fin_energy_{args.config}_{args.date}")
    outdir.mkdir(parents=True, exist_ok=True)

    all_files = list_day_files(args.config, args.date)
    if not all_files:
        print("no files, nothing to do")
        return
    print(f"{len(all_files)} files under {args.config}/{args.date}/dphi")

    meta = None
    for f in all_files[:5]:               # tolerate a bad first file or two
        try:
            meta = read_meta(f)
            break
        except Exception as e:            # noqa: BLE001
            print(f"  ! unreadable file {f.name}: {e}")
    if meta is None:
        print("no readable files, nothing to do")
        return
    print(f"fs={meta['fs']:.0f} Hz  dx={meta['dx']:.3f} m  "
          f"n_chan={meta['n_chan']}  ({meta['n_time']/meta['fs']:.0f}s/file)")

    print("reading file timestamps ...")
    with Pool(args.nproc) as pool:
        ep_raw = pool.map(file_epoch, all_files)
    keep = [(f, e) for f, e in zip(all_files, ep_raw) if e is not None]
    if len(keep) < len(all_files):
        print(f"  skipped {len(all_files) - len(keep)} unreadable file(s)")
    all_files = [f for f, _ in keep]
    epochs = np.array([e for _, e in keep])

    day0 = datetime.fromtimestamp(epochs[0], tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    s = parse_hhmm(args.start)
    e = parse_hhmm(args.end)
    keep = np.ones(len(all_files), dtype=bool)
    if s is not None:
        keep &= epochs >= day0 + s
    if e is not None:
        keep &= epochs <= day0 + e
    idx_kept = np.where(keep)[0]
    if idx_kept.size == 0:
        print("time window selects no files")
        return
    i0, i1 = int(idx_kept.min()), int(idx_kept.max())
    print(f"scanning files [{i0}:{i1+1}] "
          f"({datetime.fromtimestamp(epochs[i0], tz=timezone.utc):%H:%M} - "
          f"{datetime.fromtimestamp(epochs[i1], tz=timezone.utc):%H:%M} UTC)")

    channel_dists = np.arange(args.dist_min, args.dist_max + 1, args.dist_step)
    det_kwargs = dict(
        band=tuple(args.band),
        height_mult=args.height_mult,
        min_peaks=args.min_peaks,
        ipi_min=args.ipi[0],
        ipi_max=args.ipi[1],
        min_good_ipis=args.min_good_ipis,
        min_fin_conc=args.min_fin_conc,
        pulse_band=tuple(args.pulse_band),
        max_ipi_cv=args.max_ipi_cv,
        conc_ref_hz=tuple(args.conc_ref),
        min_fin_conc_norm=args.min_fin_conc_norm,
    )
    step = max(1, BUFFER_FILES - BUFFER_OVERLAP_FILES)
    starts = list(range(i0, i1 + 1, step))
    print(f"{len(starts)} rolling buffers x {channel_dists.size} channels, "
          f"{args.nproc} workers ...")

    csv_path = outdir / "detections.csv"
    work = partial(_worker, all_files=all_files, epochs=epochs, meta=meta,
                   channel_dists=channel_dists, det_kwargs=det_kwargs)

    # scan in chunks and checkpoint detections.csv after each, so a long run on a
    # busy box is observable and survives an interruption
    CHUNK = 40
    cands: list[Candidate] = []
    done = 0
    t_scan = time.time()
    with Pool(args.nproc) as pool:
        for ci in range(0, len(starts), CHUNK):
            batch = starts[ci:ci + CHUNK]
            for r in pool.imap_unordered(work, batch, chunksize=2):
                cands.extend(r)
                done += 1
            events = merge_candidates(cands)
            write_csv([event_row(ev, args.config, args.date) for ev in events], csv_path)
            rate = done / max(1e-6, time.time() - t_scan)
            eta = (len(starts) - done) / max(1e-6, rate)
            print(f"  buffers {done}/{len(starts)}  |  {len(cands)} candidates  "
                  f"-> {len(events)} events  |  {rate:.2f} buf/s  ETA {eta/60:.0f} min")

    events = merge_candidates(cands)
    print(f"{len(cands)} raw buffer-channel candidates -> {len(events)} merged events")
    write_csv([event_row(ev, args.config, args.date) for ev in events], csv_path)

    if args.scan_only:
        print(f"\nwrote {csv_path}  (scan-only, no panels)")
        return

    pngs: dict[int, str] = {}
    if events:
        n_render = min(len(events), args.max_events)
        print(f"rendering {n_render} review panels ...")
        jobs = [(k, events[k], args.config, args.date, all_files, epochs, meta, outdir, tuple(args.band))
                for k in range(n_render)]
        with Pool(min(args.nproc, max(1, n_render))) as pool:
            for k, png in pool.imap_unordered(_render_worker, jobs):
                pngs[k] = png
                ev = events[k]
                print(f"  [{k+1}/{len(events)}] {ev.utc:%H:%M:%S} UTC  "
                      f"ch {ev.best.dist_m:.0f} m  IPIs {ev.best.good_ipis_s}  "
                      f"fin-conc {ev.best.fin_conc}  fin-conc-norm {ev.best.fin_conc_norm}  "
                      f"SNR {ev.best.snr}  {png}")

    rows = [event_row(ev, args.config, args.date, pngs.get(k, ""))
            for k, ev in enumerate(events)]
    write_csv(rows, csv_path)

    if rows:
        write_index(events[: args.max_events], rows[: args.max_events],
                    args.config, args.date, outdir)

    print(f"\nwrote {csv_path}")
    print(f"wrote {outdir/'index.html'}  ({len(pngs)} panels)")
    print("open the index.html and confirm each event by eye "
          "(look for a clean row of equally-spaced ~20 Hz downswept pulses).")


if __name__ == "__main__":
    main()

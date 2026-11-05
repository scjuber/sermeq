"""
Batch along-cable localization (x_w) for every event in the fin-whale energy detector's
catalog (fin_energy_<config>_<date>/detections.csv, all 7 days), using the exact
matched-filter-track + apex-fit + BIC/bootstrap pipeline validated in
whale-localization.ipynb (Steps 2-4, 7) -- not the detector's crude rep_channel_m.

For each event:
  1. nearest_file_for_epoch() turns (t_epoch + first pulse offset) into a (center_time,
     ref_time_guess) pair -- the exact 10s file the seed pulse falls in, and its offset
     into that file.
  2. load_padded_band_and_ref() reads the raw channel block ONCE (aperture rep_channel_m
     +/- APERTURE_M) and derives both the bandpassed tracking array (matched_filter_track
     needs it filtered) and the raw single-reference-channel trace (repeating_bursts /
     spectrogram check needs it unfiltered) from the same read -- avoiding the two
     separate full rereads whale-localization.ipynb's Step 2 + Step 5 do serially (the
     dominant per-event cost on this filesystem is the number of file opens, not compute).
  3. matched_filter_track -> validate_apex_fit (BIC + 100-resample bootstrap) -> x_w, v,
     rms, apex_visible, CI.
  4. repeating_bursts() on the raw reference trace, as an independent real-fin-whale check.

"Successful" localization = apex_visible (BIC favors the apex model over a straight line,
AND x_w isn't pinned at the aperture edge) AND has_bursts (repeating ~15-30Hz envelope
peaks at fin-song intervals) -- the same two-gate standard used throughout
whale-localization.ipynb; n_tracked (channels the matched filter actually locked onto)
is reported too but not gating.

Runs events in parallel (I/O-latency-bound, not CPU-bound -- each worker is mostly
waiting on network file reads, so many can be in flight at once); the main process
checkpoints results to disk as they complete.

Usage:
    python batch_localize_events.py --nproc 12
    python batch_localize_events.py --nproc 12 --dates 20241108 20241112
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import (butter, correlate, detrend, find_peaks, hilbert,
                          sosfiltfilt, spectrogram)

DATA_ROOT = Path("/1-fnp/petasaur/p-jbod1/morro/Morro_SiteData")
REPO = Path("/home/sjuber/interferometry-for-orcas")
RNG = np.random.default_rng(0)

APERTURE_M = 1200.0     # +/- around rep_channel_m for the tracking window
BEFORE_S, AFTER_S = 60.0, 60.0
BAND_HZ = (10.0, 35.0)
N_BOOT = 100
DATES = ["20241106", "20241107", "20241108", "20241109", "20241110", "20241111", "20241112"]
CONFIG_FOR = {d: "MorroPhase_base" for d in DATES}


# --------------------------------------------------------------------------- #
#  file indexing (per day, cached -- filenames encode timestamps, no h5py needed)
# --------------------------------------------------------------------------- #
_DAY_FILES_CACHE: dict[str, tuple[list[Path], np.ndarray]] = {}


def dphi_dir_for(date_str: str) -> Path:
    config = "MorroPhase" if date_str == "20241105" else "MorroPhase_base"
    return DATA_ROOT / config / date_str / "dphi"


def day_files(date_str: str) -> tuple[list[Path], np.ndarray]:
    if date_str in _DAY_FILES_CACHE:
        return _DAY_FILES_CACHE[date_str]
    d = dphi_dir_for(date_str)
    all_files = sorted(d.glob("*.hdf5"))
    times = np.array([
        datetime.strptime(date_str + f.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        for f in all_files
    ])
    _DAY_FILES_CACHE[date_str] = (all_files, times)
    return all_files, times


def nearest_file_for_epoch(date_str: str, epoch: float) -> tuple[str, float]:
    all_files, times = day_files(date_str)
    idx = np.searchsorted(times, epoch, side="right") - 1
    idx = max(0, min(idx, len(all_files) - 1))
    return all_files[idx].stem, epoch - times[idx]


def files_around(date_str: str, center_time_str: str, before_s: float, after_s: float,
                 pad_files: int = 1) -> tuple[list[Path], list[Path], list[Path]]:
    all_files, nominal = day_files(date_str)
    center = datetime.strptime(date_str + center_time_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
    mask = (nominal >= center - before_s - 1) & (nominal <= center + after_s + 1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"no files around {date_str} {center_time_str}")
    i0, i1 = idx.min(), idx.max()
    i0p, i1p = max(0, i0 - pad_files), min(len(all_files) - 1, i1 + pad_files)
    return all_files[i0p:i0], all_files[i0:i1 + 1], all_files[i1 + 1:i1p + 1]


# --------------------------------------------------------------------------- #
#  combined load: one read -> both the bandpassed tracking array and the raw
#  single-reference-channel trace (avoids load_padded_band + spectrogram_at's
#  separate rereads of the same files)
# --------------------------------------------------------------------------- #
def load_padded_band_and_ref(date_str, center_file, pad_before, analysis_files, pad_after,
                             dist_range_m, ref_dist_m, band=BAND_HZ, ref_stack_half_m=8.0):
    all_files = pad_before + analysis_files + pad_after
    with h5py.File(analysis_files[0], "r") as fp:
        fs = 1 / fp["header/dt"][()]
        dx = fp["header/dx"][()]
        scale = fp["header/dataScale"][()]
        ch0 = max(0, int(round(dist_range_m[0] / dx)))
        ch1 = min(fp["data"].shape[1], int(round(dist_range_m[1] / dx)))

    ref_ch = int(round(ref_dist_m / dx))
    ref_half = max(1, int(round(ref_stack_half_m / dx)))
    ref_lo, ref_hi = ref_ch - ref_half, ref_ch + ref_half + 1
    # ref channel window must be a sub-range of [ch0, ch1) to slice out of the same read
    ref_lo_local, ref_hi_local = max(ref_lo, ch0) - ch0, min(ref_hi, ch1) - ch0

    blocks, times = [], []
    for f in all_files:
        with h5py.File(f, "r") as fp:
            blocks.append(fp["data"][:, ch0:ch1].astype(np.float32) * scale)
            times.append(fp["header/time"][()])
    Xall = np.vstack(blocks)
    t_start_pad = times[0]
    t_start_analysis = times[len(pad_before)]

    ref_raw = detrend(Xall[:, ref_lo_local:ref_hi_local].mean(axis=1))

    Xall_d = detrend(Xall, axis=0)
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, Xall_d, axis=0)

    trim0 = int(round((t_start_analysis - t_start_pad) * fs))
    n_analysis = int(round(len(analysis_files) * 10 * fs))
    X = Xf[trim0: trim0 + n_analysis]
    ref_raw = ref_raw[trim0: trim0 + n_analysis]

    with h5py.File(center_file, "r") as fp:
        t_center = fp["header/time"][()]
    t_offset = t_start_analysis - t_center
    t_axis = np.arange(X.shape[0]) / fs + t_offset
    x_positions = (ch0 + np.arange(X.shape[1])) * dx
    return X, t_axis, x_positions, fs, dx, ref_raw


# --------------------------------------------------------------------------- #
#  matched-filter track + apex fit + BIC/bootstrap (verbatim from
#  whale-localization.ipynb Steps 2-3, factored out for reuse)
# --------------------------------------------------------------------------- #
def matched_filter_track(X, t_axis, x_positions, dx, ref_dist, ref_time_guess, fs,
                         template=None, template_half_s=0.06,
                         seed_half_win_s=1.0, track_win_s=0.3, step=3):
    ref_ch = int(round((ref_dist - x_positions[0]) / dx))
    if template is None:
        i0 = max(0, np.searchsorted(t_axis, ref_time_guess - template_half_s))
        i1 = min(len(t_axis), np.searchsorted(t_axis, ref_time_guess + template_half_s))
        template = X[i0:i1, ref_ch].copy()
    tmpl = template - template.mean()
    tmpl_energy = np.sum(tmpl ** 2)

    def local_match(ch, t_guess, half_win_s):
        i0 = max(0, np.searchsorted(t_axis, t_guess - half_win_s))
        i1 = min(len(t_axis), np.searchsorted(t_axis, t_guess + half_win_s))
        if i1 - i0 < len(tmpl):
            return None, None
        seg = X[i0:i1, ch]
        seg_c = seg - seg.mean()
        xc = correlate(seg_c, tmpl, mode="valid")
        norm = np.sqrt(tmpl_energy * np.sum(seg_c ** 2) + 1e-30)
        xc_n = xc / norm
        k = np.argmax(np.abs(xc_n))
        return t_axis[i0 + k] + (len(tmpl) / 2) / fs, np.abs(xc_n[k])

    n_ch = X.shape[1]
    picked_t, picked_a = np.full(n_ch, np.nan), np.full(n_ch, np.nan)
    t_seed, a_seed = local_match(ref_ch, ref_time_guess, seed_half_win_s)
    picked_t[ref_ch], picked_a[ref_ch] = t_seed, a_seed

    prev_t = t_seed
    for ch in range(ref_ch + step, n_ch, step):
        t_pick, a_pick = local_match(ch, prev_t, track_win_s)
        if t_pick is None:
            continue
        picked_t[ch], picked_a[ch] = t_pick, a_pick
        prev_t = t_pick
    prev_t = t_seed
    for ch in range(ref_ch - step, -1, -step):
        t_pick, a_pick = local_match(ch, prev_t, track_win_s)
        if t_pick is None:
            continue
        picked_t[ch], picked_a[ch] = t_pick, a_pick
        prev_t = t_pick

    valid = ~np.isnan(picked_t)
    return picked_t, picked_a, valid


def fit_apex(x, t, t_axis, x_positions):
    def resid(params, x, t):
        t0, x_w, v = params
        return t0 + np.abs(x - x_w) / v - t
    x0 = [t.min(), np.median(x), 300.0]
    res = least_squares(resid, x0, args=(x, t),
                        bounds=([t_axis[0], x_positions.min(), 30],
                                [t_axis[-1], x_positions.max(), 3300]))
    rss = np.sum(res.fun ** 2)
    return dict(t0=res.x[0], x_w=res.x[1], v=res.x[2], rms_s=np.sqrt(rss / len(x)), rss=rss, n=len(x))


def fit_straight_line(x, t):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    a, b = coef
    return dict(rss=np.sum((a + b * x - t) ** 2), n=len(x))


def bic(rss, n, k):
    return n * np.log(max(rss, 1e-30) / n) + k * np.log(n)


def validate_apex_fit(x, t, t_axis, x_positions, n_boot=N_BOOT, rng=RNG):
    fit = fit_apex(x, t, t_axis, x_positions)
    straight = fit_straight_line(x, t)
    bic_apex = bic(fit["rss"], fit["n"], k=3)
    bic_straight = bic(straight["rss"], straight["n"], k=2)

    n = len(x)
    x_ws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            x_ws[b] = fit_apex(x[idx], t[idx], t_axis, x_positions)["x_w"]
        except Exception:
            x_ws[b] = np.nan
    x_w_ci = np.nanpercentile(x_ws, [2.5, 97.5])

    margin = 0.02 * (x_positions.max() - x_positions.min())
    apex_visible = (fit["x_w"] > x_positions.min() + margin
                   and fit["x_w"] < x_positions.max() - margin
                   and bic_apex < bic_straight)
    return dict(fit=fit, bic_apex=bic_apex, bic_straight=bic_straight,
               x_w_ci=x_w_ci, apex_visible=apex_visible)


def repeating_bursts(trace, fs, band=(15, 30), min_interval_s=8, max_interval_s=30):
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, trace)))
    baseline = np.median(env)
    peaks, _ = find_peaks(env, height=4 * baseline, distance=int(min_interval_s * fs * 0.5))
    if len(peaks) < 2:
        return False
    intervals = np.diff(peaks) / fs
    return bool(np.mean((intervals >= min_interval_s) & (intervals <= max_interval_s)) >= 0.5)


# --------------------------------------------------------------------------- #
#  per-event driver
# --------------------------------------------------------------------------- #
def localize_event(config: str, date: str, utc: str, t_epoch: float, rep_channel_m: float,
                   peak_times_s: list[float]) -> dict:
    base = dict(config=config, date=date, utc=utc, rep_channel_m=rep_channel_m)
    try:
        seed_epoch = t_epoch + (peak_times_s[0] if peak_times_s else 0.0)
        center_time, ref_time_guess = nearest_file_for_epoch(date, seed_epoch)
        center_file = dphi_dir_for(date) / f"{center_time}.hdf5"
        dist_range_m = (max(0.0, rep_channel_m - APERTURE_M), rep_channel_m + APERTURE_M)
        pad_before, analysis_files, pad_after = files_around(date, center_time, BEFORE_S, AFTER_S)

        X, t_axis, x_positions, fs, dx, ref_raw = load_padded_band_and_ref(
            date, center_file, pad_before, analysis_files, pad_after, dist_range_m, rep_channel_m)

        picked_t, picked_a, valid = matched_filter_track(
            X, t_axis, x_positions, dx, rep_channel_m, ref_time_guess, fs)
        n_tracked = int(valid.sum())
        if n_tracked < 10:
            return dict(**base, n_tracked=n_tracked, x_w=np.nan, v=np.nan, rms_ms=np.nan,
                       x_w_ci_lo=np.nan, x_w_ci_hi=np.nan, apex_visible=False,
                       has_bursts=False, localized=False, note="too few channels tracked")

        result = validate_apex_fit(x_positions[valid], picked_t[valid], t_axis, x_positions)
        fit = result["fit"]
        has_bursts = repeating_bursts(ref_raw, fs)
        localized = bool(result["apex_visible"] and has_bursts)

        return dict(**base, n_tracked=n_tracked, x_w=round(float(fit["x_w"]), 1),
                   v=round(float(fit["v"]), 1), rms_ms=round(float(fit["rms_s"]) * 1000, 1),
                   x_w_ci_lo=round(float(result["x_w_ci"][0]), 1),
                   x_w_ci_hi=round(float(result["x_w_ci"][1]), 1),
                   bic_apex=round(float(result["bic_apex"]), 1),
                   bic_straight=round(float(result["bic_straight"]), 1),
                   apex_visible=bool(result["apex_visible"]), has_bursts=has_bursts,
                   localized=localized, note="")
    except Exception as e:
        return dict(**base, n_tracked=0, x_w=np.nan, v=np.nan, rms_ms=np.nan,
                   x_w_ci_lo=np.nan, x_w_ci_hi=np.nan, apex_visible=False,
                   has_bursts=False, localized=False, note=f"error: {e}")


def _worker(args) -> dict:
    config, date, utc, t_epoch, rep_channel_m, peak_times_s = args
    return localize_event(config, date, utc, t_epoch, rep_channel_m, peak_times_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--dates", nargs="+", default=DATES)
    ap.add_argument("--out", default=str(REPO / "results/localization_results/data/event_localization.csv"))
    args = ap.parse_args()

    jobs = []
    for date in args.dates:
        config = CONFIG_FOR.get(date, "MorroPhase_base")
        det_csv = REPO / "results" / f"fin_energy_{config}_{date}" / "detections.csv"
        if not det_csv.exists():
            print(f"  no detections.csv for {config}/{date}, skipping")
            continue
        with det_csv.open() as fh:
            for row in csv.DictReader(fh):
                peak_times = [float(x) for x in row["peak_times_s"].split("|") if x]
                jobs.append((config, date, row["utc"], float(row["t_epoch"]),
                            float(row["rep_channel_m"]), peak_times))
    print(f"{len(jobs)} events queued across {len(args.dates)} day(s), {args.nproc} workers")

    # pre-warm the per-day file index in the main process before forking, so workers
    # inherit it (fork start method on Linux) instead of every worker re-globbing/
    # re-parsing the same ~8600 filenames per day redundantly
    for date in set(j[1] for j in jobs):
        day_files(date)

    out_path = Path(args.out)
    results: list[dict] = []
    from multiprocessing import Pool
    with Pool(args.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
            results.append(r)
            flag = "OK" if r["localized"] else ("--" if not r["note"] else "!!")
            print(f"  [{i}/{len(jobs)}] {r['date']} {r['utc']}  x_w={r['x_w']}m  "
                  f"apex_visible={r.get('apex_visible')}  bursts={r.get('has_bursts')}  "
                  f"-> {flag} {r['note']}")
            if i % 5 == 0 or i == len(jobs):
                with out_path.open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
                    w.writeheader(); w.writerows(results)

    print(f"\nwrote {len(results)} rows -> {out_path}")
    n_ok = sum(1 for r in results if r["localized"])
    print(f"localized (apex_visible AND repeating bursts): {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()

"""
Export each DAS fin-whale detection in fin_whale_detections_combined.csv to its own
WAV file, for annotation in external software (Raven, Audacity, etc.).

Reuses fin-whale-energy-detector.py's own file-discovery and channel-stacking logic
(list_day_files, file_epoch, read_meta, read_span, STACK_HALF_M) so each clip is
built the same way the detector's own review-panel representative trace is -- same
representative channel, same +/-12m stack.

Usage:
    python export_detections_to_wav.py
"""
import csv
import importlib.util
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import detrend

REPO = Path("/home/sjuber/interferometry-for-orcas")
OUTDIR = REPO / "results" / "wav_detections"
OUTDIR.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("det", str(REPO / "fin-whale-energy-detector.py"))
det = importlib.util.module_from_spec(spec)
sys.modules["det"] = det
spec.loader.exec_module(det)

MAX_DURATION_S = 250.0   # matches the review-panel cap (PANEL_MAX_SPAN_S)
PAD_S = 15.0


def export_row(r, file_cache):
    config, date = r["config"], r["date"]
    t_epoch = float(r["t_epoch"])
    rep_m = float(r["rep_channel_m"])
    peak_times = [float(x) for x in r["peak_times_s"].split("|")] if r["peak_times_s"] else [120.0]
    duration = min(max(peak_times) + PAD_S, MAX_DURATION_S)

    key = (config, date)
    if key not in file_cache:
        all_files = det.list_day_files(config, date)
        with Pool(16) as pool:                      # parallel header reads -- sequential
            ep_raw = pool.map(det.file_epoch, all_files)   # was ~15-30min/day over /1-fnp
        keep = [e is not None for e in ep_raw]
        file_cache[key] = (
            [f for f, k in zip(all_files, keep) if k],
            np.array([e for e, k in zip(ep_raw, keep) if k], dtype=float),
        )
    all_files, epochs = file_cache[key]

    lo = int(np.argmin(np.abs(epochs - t_epoch)))
    hi = lo
    while hi < len(all_files) - 1 and epochs[hi] - epochs[lo] < duration:
        hi += 1
    files = all_files[lo:hi + 1]

    meta = det.read_meta(files[0])
    fs, dx, scale = meta["fs"], meta["dx"], meta["scale"]
    c = int(round(rep_m / dx))
    stack_half = max(1, int(round(det.STACK_HALF_M / dx)))
    cols = np.arange(max(0, c - stack_half), min(meta["n_chan"], c + stack_half + 1))
    span, ch_lo = det.read_span(files, cols, scale)
    trace = span[:, cols - ch_lo].mean(axis=1)
    trace = detrend(trace)
    peak = np.max(np.abs(trace))
    trace_norm = (trace / peak * 0.9).astype(np.float32) if peak > 0 else trace.astype(np.float32)

    utc_dt = datetime.fromtimestamp(t_epoch, tz=timezone.utc)
    fname = f"{config}_{date}_{utc_dt:%H%M%S}_ch{rep_m:.0f}m.wav"
    sf.write(OUTDIR / fname, trace_norm, int(round(fs)))
    return fname, len(trace_norm) / fs


def main():
    rows = [r for r in csv.DictReader(open(REPO / "results" / "fin_whale_detections_combined.csv"))
            if r["source"] == "fin_energy_detector"]
    print(f"{len(rows)} detections to export")
    file_cache = {}
    ok, failed = 0, []
    for i, r in enumerate(rows):
        try:
            fname, dur = export_row(r, file_cache)
            ok += 1
            print(f"  [{i+1}/{len(rows)}] {fname}  ({dur:.0f}s)", flush=True)
        except Exception as e:                                   # noqa: BLE001
            failed.append((r["utc"], str(e)))
            print(f"  [{i+1}/{len(rows)}] FAILED {r['utc']}: {e}", flush=True)
    print(f"\n{ok}/{len(rows)} exported to {OUTDIR}")
    if failed:
        print(f"{len(failed)} failed:")
        for utc, err in failed:
            print(f"  {utc}: {err}")


if __name__ == "__main__":
    main()

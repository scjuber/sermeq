"""
Per-day survey of Morro Bay DAS data: an FX (frequency-distance) plot and an
animated FX-heatmap "spectrogram video" for each recording day.

Why partial reads: the data mount caps out around ~100-150MB/s no matter how
many parallel readers are used (benchmarked directly), so total bytes read is
what determines runtime. Reading only the first `SAMPLES_PER_FILE` samples of
each 10s file (instead of all 5000) cuts I/O ~5x while still touching every
file, so the video keeps full time cadence across the day. Channel selection
does NOT reduce read time (storage is row-major/contiguous per time sample,
so a full row must be fetched regardless of which channels are requested) --
only the samples/time axis does.

Why manual frame compositing instead of matplotlib's per-frame canvas.draw():
redrawing a full 9002-row imshow + colorbar every frame cost ~0.58s/frame
(more than I/O). Instead we render the static axes/labels/colorbar once
("chrome"), then per frame: block-average the channel axis down to display
resolution, map through a fixed colormap (numpy, vectorized), and paste the
resulting tile into a copy of the chrome image with PIL. That's ~25x faster
and gives an identical-looking result since 9002 unique rows can't be
resolved in a ~500px-tall video anyway. The FX plot (static, one per day)
and the underlying PSD averaging still use the full, non-downsampled data.

Usage:
    python das_survey.py --date 20241106
    python das_survey.py --date 20241106 --max-files 100   # quick test
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import welch

BASE = Path("/1-fnp/petasaur/p-jbod1/morro/Morro_SiteData/MorroPhase_base")
OUT_ROOT = Path("/home/sjuber/interferometry-for-orcas/survey_output")

SAMPLES_PER_FILE = 1024   # ~2.05s snippet read from the start of each 10s file
NPERSEG = 256              # welch segment length -> ~1.95 Hz resolution
IO_WORKERS = 8
CHAN_STEP_FX = 5           # FX plot uses every 5th channel per the request
VIDEO_FPS = 24
PROBE_FILE_COUNT = 20      # files sampled up front to set a stable color scale
FIGSIZE = (10, 7)
DPI = 100

# Channels beyond this are a hardware artifact at the physical end of the
# fiber: RMS jumps from ~2 to ~100+ right around channel 8750 (verified
# across multiple dates/times), not real cable signal. Truncated out so it
# doesn't dominate the color scale.
N_VALID_CHANNELS = 8740


def day_files(date: str, base: Path = BASE):
    folder = base / date / "dphi"
    return sorted(folder.glob("*.hdf5"))


def read_raw(fname, n_samples=SAMPLES_PER_FILE):
    try:
        with h5py.File(fname, "r") as fp:
            scale = fp["header/dataScale"][()]
            t = fp["header/time"][()] if "header/time" in fp else None
            raw = fp["data"][:n_samples, :N_VALID_CHANNELS].astype(np.float32)
        return raw * scale, t
    except OSError as e:
        print(f"[warn] skipping unreadable file {fname}: {e}")
        return None, None


def compute_psd(raw, fs):
    f, Pxx = welch(raw, fs=fs, nperseg=NPERSEG, noverlap=0, axis=0, detrend="constant")
    return f, Pxx  # Pxx: (n_freq, n_channels), linear power


def probe_scale(files, fs):
    """Sample a few files up front to pick a stable dB color range for the whole day."""
    sample = files[:: max(1, len(files) // PROBE_FILE_COUNT)][:PROBE_FILE_COUNT]
    vals = []
    for fname in sample:
        raw, _ = read_raw(fname)
        if raw is None:
            continue
        _, Pxx = compute_psd(raw, fs)
        vals.append(10 * np.log10(Pxx + 1e-20))
    vals = np.concatenate(vals, axis=1)
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 99))


def build_chrome(fs, n_channels, dx, vmin, vmax):
    """Render the static axes/labels/colorbar once; return (chrome_img, paste_rect, video_wh)."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    n_freq = NPERSEG // 2 + 1
    im = ax.imshow(
        np.zeros((n_channels, n_freq)),
        aspect="auto",
        extent=[0, fs / 2, n_channels * dx, 0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Optical distance (m)")
    plt.colorbar(im, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    w, h = fig.canvas.get_width_height()
    chrome = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    x0, y0, x1, y1 = bbox.extents
    col0, col1 = int(round(x0)), int(round(x1))
    row0, row1 = int(round(h - y1)), int(round(h - y0))

    # ffmpeg (libx264) requires even dimensions
    vid_w, vid_h = (w // 2) * 2, (h // 2) * 2
    return Image.fromarray(chrome), (col0, row0, col1, row1), (vid_w, vid_h)


def process_day(date: str, max_files=None, log_every=250, base_dir: Path = BASE):
    files = day_files(date, base=base_dir)
    if not files:
        raise FileNotFoundError(f"No files found for {date}")
    if max_files:
        files = files[:max_files]

    with h5py.File(files[0], "r") as fp:
        fs = 1.0 / fp["header/dt"][()]
        dx = fp["header/dx"][()]
        n_channels = min(fp["data"].shape[1], N_VALID_CHANNELS)

    out_dir = OUT_ROOT / date
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{date}] {len(files)} files, fs={fs}Hz, dx={dx:.4f}m, n_channels={n_channels}")
    print(f"[{date}] probing color scale...")
    vmin, vmax = probe_scale(files, fs)
    print(f"[{date}] dB range for color scale: [{vmin:.1f}, {vmax:.1f}]")

    fx_channels = np.arange(0, n_channels, CHAN_STEP_FX)
    fx_distance = fx_channels * dx
    n_freq = NPERSEG // 2 + 1
    power_sum = np.zeros((n_freq, n_channels), dtype=np.float64)

    chrome_img, (col0, row0, col1, row1), (vid_w, vid_h) = build_chrome(fs, n_channels, dx, vmin, vmax)
    target_rows, target_cols = row1 - row0, col1 - col0
    bin_sz = max(1, n_channels // target_rows)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    video_path = out_dir / f"spectrogram_video_{date}.mp4"
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264", quality=8)

    t0 = time.time()
    n_done = 0
    first_time = None
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        for raw, t in ex.map(read_raw, files):
            if raw is None:
                continue
            if first_time is None and t is not None:
                first_time = t
            _, Pxx = compute_psd(raw, fs)
            power_sum += Pxx

            Pxx_db = 10 * np.log10(Pxx.T + 1e-20)  # (n_channels, n_freq)
            trimmed = Pxx_db[: target_rows * bin_sz]
            ds = trimmed.reshape(target_rows, bin_sz, -1).mean(axis=1)
            rgb = (cmap(norm(ds))[:, :, :3] * 255).astype(np.uint8)
            tile = Image.fromarray(rgb).resize((target_cols, target_rows), Image.BILINEAR)

            frame_img = chrome_img.copy()
            frame_img.paste(tile, (col0, row0))
            draw = ImageDraw.Draw(frame_img)
            label = f"{date}  t={n_done * 10}s"
            if t is not None:
                label += f"  ({time.strftime('%H:%M:%S', time.gmtime(t))} UTC)"
            draw.text((col0 + 5, row0 + 5), label, fill=(255, 255, 255))

            frame = np.asarray(frame_img)[:vid_h, :vid_w]
            writer.append_data(frame)

            n_done += 1
            if n_done % log_every == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(files) - n_done) / rate if rate > 0 else float("nan")
                print(
                    f"[{date}] {n_done}/{len(files)} files "
                    f"({rate:.2f} files/s, ETA {eta/60:.1f} min)"
                )

    writer.close()

    avg_psd_db = 10 * np.log10(power_sum / n_done + 1e-20)  # (n_freq, n_channels)
    fx_grid = avg_psd_db[:, fx_channels]  # (n_freq, n_fx_channels)

    plt.rcParams.update({"font.size": 14})
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(
        fx_grid.T,
        aspect="auto",
        origin="upper",
        extent=[0, fs / 2, fx_distance[-1], fx_distance[0]],
        cmap="viridis",
        vmin=np.percentile(fx_grid, 5),
        vmax=np.percentile(fx_grid, 99),
    )
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Optical distance (m)")
    ax2.set_title(f"Day-averaged FX plot — {date} (every {CHAN_STEP_FX}th channel)")
    plt.colorbar(im2, ax=ax2, label="PSD (dB)")
    fig2.tight_layout()
    fx_path = out_dir / f"fx_plot_{date}.png"
    fig2.savefig(fx_path, dpi=150)
    plt.close(fig2)

    meta = {
        "date": date,
        "n_files_processed": n_done,
        "fs_hz": fs,
        "dx_m": dx,
        "n_channels": n_channels,
        "samples_per_file": SAMPLES_PER_FILE,
        "nperseg": NPERSEG,
        "chan_step_fx": CHAN_STEP_FX,
        "video_fps": VIDEO_FPS,
        "vmin_db": vmin,
        "vmax_db": vmax,
        "wall_time_s": time.time() - t0,
        "first_file_time_utc": first_time,
        "video_path": str(video_path),
        "fx_plot_path": str(fx_path),
    }
    with open(out_dir / f"meta_{date}.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[{date}] done in {meta['wall_time_s']/60:.1f} min -> {video_path}, {fx_path}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="e.g. 20241106")
    parser.add_argument("--max-files", type=int, default=None, help="limit files for a quick test")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE,
        help="root data folder containing <date>/dphi/*.hdf5 (default: MorroPhase_base)",
    )
    args = parser.parse_args()
    process_day(args.date, max_files=args.max_files, base_dir=args.base_dir)

# Fin whale detection and localization from DAS on the Morro Bay cable

Detecting and characterizing fin whale calls using Distributed Acoustic Sensing (DAS)
on the Morro Bay submarine telecom cable, with cross-checks against a nearby
hydrophone, AIS vessel tracks, and human (Raven) annotation.

**Start here:** [`notebooks/call_metrics_analysis.ipynb`](notebooks/call_metrics_analysis.ipynb)
— the finished analysis. For 785 Raven-confirmed 20 Hz calls across 122 clips (~4.4 h
of audio), it computes duration, peak/center frequency, bandwidth, SNR, inter-pulse
interval (with an SNR gate to avoid pairing calls from different callers or multipath
reflections), and song-type classification (singlet/doublet/triplet).

## Folder map

| Folder | What's in it |
|---|---|
| `notebooks/` | Active notebooks: the call-metrics analysis above, `cable_maps.ipynb` (site map figures), and the cross-correlation template-matching chain (`make-templates.ipynb` → `fast-xcorr.py` → `read-xc-results.ipynb`), plus `morro-bay.ipynb`. |
| `scripts/` | The detector (`fin-whale-energy-detector.py`) and the pipeline scripts that feed off its output — WAV export, spectrogram annotation, localization, gallery/figure generation, and `_build_call_metrics_notebook.py` (regenerates the main notebook's cell source; re-run this, not the notebook file directly, after editing its logic). `cable_geo.py` and `das_survey.py` are shared dependencies. |
| `results/` | Generated output: per-day detector runs (`fin_energy_MorroPhase_base_<date>/`, one per 2024-11-06 through 2024-11-12), the combined detections CSV, exported/annotated call WAVs, and `localization_results/` (position/velocity fits, figures, the call-metrics CSVs). Nothing here is hand-written — regenerate it by re-running the scripts. |
| `data/` | Supporting input data this pipeline reads but doesn't generate: hydrophone cross-check clips, nautical chart data for cable geolocation, a sample audio file, cable depth reference data. |
| `docs/` | Handover notes — read [`docs/fin_whale_detector_handover.md`](docs/fin_whale_detector_handover.md) for how the detector works, and [`docs/HANDOVER_restart_elsewhere.md`](docs/HANDOVER_restart_elsewhere.md) if you're picking this up on a new machine (environment setup, where the raw DAS data lives, known issues). |
| `archive/` | Earlier research phases and investigations, kept intact but out of the way: pre-fin-whale DAS exploration (`archive/notebooks/`), four different beamforming/localization approaches tried and compared (`archive/beamforming_investigation/`), six iterations of a precursor-call search (`archive/precursor_search/`), humpback/other-species detector work (`archive/other_species/`), the stalled 40 Hz call detection effort (`archive/fin_energy_40Hz/`), and operational logs/backups/driver scripts (`archive/logs_and_drivers/`, `archive/backups/`). |
| `handoff_received_level/` | A self-contained task package (brief + starter script) for adding calibrated received-level amplitude to the call metrics, written for whoever has working access to the raw DAS data — see its own `HANDOFF.md`. |

## Before running anything

- **Raw DAS data**: the detector and WAV-export scripts read from a path hardcoded as
  `DATA_ROOT` in `scripts/fin-whale-energy-detector.py` (default
  `/1-fnp/petasaur/p-jbod1/morro/Morro_SiteData`). This is an NFS mount that goes down
  periodically — check `docs/HANDOVER_restart_elsewhere.md` before assuming it's
  reachable. The notebook in `notebooks/call_metrics_analysis.ipynb` does **not** need
  it — it only reads already-exported files under `results/wav_detections/`.
- **Python environment**: see `docs/HANDOVER_restart_elsewhere.md` for the package
  list. `.venv/` may not be portable across machines (symlinks to a specific system
  Python) — rebuild it rather than assuming it works as-is after a copy.

"""
Real-world geo-referencing for the Morro Bay DAS cable.

Factored out of whale-localization.ipynb's "Real cable geometry -- NOAA ENC,
reproducible" section so any script/notebook can convert along-fiber distance to
lon/lat and distance offshore without re-deriving the cable route. Uses the cached
ENC download in enc_data/ (NOAA ENC cell US3CA85M, CBLSUB layer); validated there to
within 0.1m against two independent reference points (KP0 landing, KP10).

Usage:
    from cable_geo import channel_to_lonlat, distance_offshore_m
    lon, lat = channel_to_lonlat(8100.0)
    d_off = distance_offshore_m(8100.0)
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/home/sjuber/interferometry-for-orcas")
ENC_DIR = REPO / "data" / "enc_data"
LANDING_LONLAT = np.array([-120.876144, 35.301745])   # KP0, independently known

_cable = None  # lazily-populated dict: cumdist, coords, m_per_deg_lon/lat


def _load_cable() -> dict:
    global _cable
    if _cable is not None:
        return _cable

    import geopandas as gpd
    from shapely.geometry import LineString

    ENC_DIR.mkdir(exist_ok=True)
    enc_zip = ENC_DIR / "US3CA85M.zip"
    if not enc_zip.exists():
        subprocess.run(["curl", "-sL", "https://charts.noaa.gov/ENCs/US3CA85M.zip",
                        "-o", str(enc_zip)], check=True)
        z = zipfile.ZipFile(enc_zip)
        assert z.testzip() is None, "corrupted ENC download -- retry"
        z.extractall(ENC_DIR)

    enc_path = ENC_DIR / "ENC_ROOT" / "US3CA85M" / "US3CA85M.000"
    cables = gpd.read_file(enc_path, layer="CBLSUB")

    def orient_from_landing(line):
        coords = np.array(line.coords)
        if np.linalg.norm(coords[-1] - LANDING_LONLAT) < np.linalg.norm(coords[0] - LANDING_LONLAT):
            coords = coords[::-1]
        return LineString(coords)

    cables["geometry"] = cables.geometry.apply(orient_from_landing)

    best_idx, best_diff = None, np.inf
    for idx, row in cables.iterrows():
        coords = np.array(row.geometry.coords)
        lat_mean = np.mean(coords[:, 1])
        dx_m = (coords[1:, 0] - coords[:-1, 0]) * 111000 * np.cos(np.radians(lat_mean))
        dy_m = (coords[1:, 1] - coords[:-1, 1]) * 111000
        length = np.sqrt(dx_m ** 2 + dy_m ** 2).sum()
        diff = abs(length - 132594.2)   # known route length, from prior bathymetry work
        if diff < best_diff:
            best_idx, best_diff = idx, diff

    route = cables.loc[best_idx].geometry
    coords = np.array(route.coords)
    lat_mean = np.mean(coords[:, 1])
    m_per_deg_lon = 111000 * np.cos(np.radians(lat_mean))
    m_per_deg_lat = 111000.0

    dx_m = (coords[1:, 0] - coords[:-1, 0]) * m_per_deg_lon
    dy_m = (coords[1:, 1] - coords[:-1, 1]) * m_per_deg_lat
    cumdist = np.concatenate([[0], np.cumsum(np.sqrt(dx_m ** 2 + dy_m ** 2))])

    _cable = dict(cumdist=cumdist, coords=coords, m_per_deg_lon=m_per_deg_lon, m_per_deg_lat=m_per_deg_lat)
    return _cable


def channel_to_lonlat(dist_along_fiber_m):
    """Along-fiber distance (m) -> (lon, lat), interpolated along the real cable route."""
    c = _load_cable()
    d = np.atleast_1d(dist_along_fiber_m)
    lon = np.interp(d, c["cumdist"], c["coords"][:, 0])
    lat = np.interp(d, c["cumdist"], c["coords"][:, 1])
    return (float(lon[0]), float(lat[0])) if np.isscalar(dist_along_fiber_m) else (lon, lat)


def channel_to_local_xy(dist_along_fiber_m, ref_lonlat=None):
    """Along-fiber distance (m) -> local Cartesian (east_m, north_m), equirectangular
    projection relative to ref_lonlat (default: the cable landing point), using the
    route's own locally-accurate m_per_deg scale factors. For 2D array-geometry
    beamforming: the cable has a real, documented ~31 deg bend (see
    distance_offshore_m), so a straight-line along-fiber-distance model breaks down
    past it -- these (east, north) coordinates carry the actual bend."""
    c = _load_cable()
    ref = np.asarray(ref_lonlat if ref_lonlat is not None else LANDING_LONLAT, dtype=float)
    lon, lat = channel_to_lonlat(dist_along_fiber_m)
    lon = np.atleast_1d(lon).astype(float)
    lat = np.atleast_1d(lat).astype(float)
    east = (lon - ref[0]) * c["m_per_deg_lon"]
    north = (lat - ref[1]) * c["m_per_deg_lat"]
    if np.isscalar(dist_along_fiber_m):
        return float(east[0]), float(north[0])
    return east, north


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_offshore_m(dist_along_fiber_m):
    """Great-circle distance (m) from the cable landing point (KP0) to the point on the
    real cable route at `dist_along_fiber_m`. This is distance from the LANDING POINT,
    not perpendicular distance to the coastline -- the cable is not straight (see the
    ~31 deg bend documented in whale-localization.ipynb), so for the first several km,
    while the route still heads roughly offshore, this tracks "how far out to sea" the
    channel is; once the route bends to run along-shelf it no longer means quite the
    same thing, so treat it as a monotonic along-route proxy, not literal shore-normal
    distance, especially past the bend."""
    lon, lat = channel_to_lonlat(dist_along_fiber_m)
    return haversine_m(LANDING_LONLAT[0], LANDING_LONLAT[1], lon, lat)


if __name__ == "__main__":
    for d in [0, 2000, 5000, 8100, 10000]:
        lon, lat = channel_to_lonlat(d)
        print(f"dist_along_fiber={d:6d}m  lon={lon:.6f}  lat={lat:.6f}  "
              f"distance_offshore={distance_offshore_m(d):7.0f}m")

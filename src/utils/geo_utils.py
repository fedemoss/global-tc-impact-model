"""Shared geospatial helpers used across feature processors."""
from __future__ import annotations

from shapely.geometry import Polygon

# EPSG:6933 — World Cylindrical Equal Area (meters). Suitable for global
# length/area calculations where local UTM zones are impractical.
GLOBAL_METRIC_EPSG = 6933


def adjust_longitude(polygon: Polygon) -> Polygon:
    """Wrap polygon longitudes from [0, 360) into [-180, 180].

    Some upstream raster sources (e.g. SRTM seamless mosaics, GPM/IMERG)
    are stored on a [-180, 180] grid, while certain GADM-derived polygons
    crossing the antimeridian carry coordinates above 180. This helper
    rewraps any such longitudes so polygon coordinates align with the
    raster CRS used downstream.
    """
    coords = list(polygon.exterior.coords)
    needs_fix = any(lon > 180 for lon, _ in coords)
    if not needs_fix:
        return polygon
    fixed = [(lon - 360 if lon > 180 else lon, lat) for lon, lat in coords]
    return Polygon(fixed)


def is_antimeridian_crossing(polygon: Polygon) -> bool:
    """Return True if the polygon straddles the meridian after wrapping."""
    coords = list(polygon.exterior.coords)
    has_pos = any(lon > 0 for lon, _ in coords)
    has_neg = any(lon < 0 for lon, _ in coords)
    return has_pos and has_neg

from __future__ import annotations

from core.config import Zone


def point_in_polygon(x: float, y: float, polygon: list[list[int]]) -> bool:
    """Ray-casting algorithm. Returns True if (x, y) is inside the polygon."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def assign_zone(cx: float, cy: float, zones: list[Zone]) -> str:
    """
    Return the name of the first zone containing point (cx, cy).

    zones = []  → full-frame camera, every detection tagged "full_frame"
    zones set, no match → detection outside all polygons, tagged "unzoned"

    Zones evaluated in config order — first match wins.
    Pass bottom-center (cx, y2) for persons so zone membership is based on
    where feet touch the ground; pass bounding-box center for everything else.
    """
    if not zones:
        return "full_frame"

    for zone in zones:
        if point_in_polygon(cx, cy, zone.polygon):
            return zone.name

    return "unzoned"

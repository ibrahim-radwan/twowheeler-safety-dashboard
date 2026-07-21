"""
figures.py — Plotly figure builders.
Each function takes data objects and returns a go.Figure.
No Dash imports here; purely Plotly.
"""

import math
import os
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional

from data import (
    TrackedObject, RISK_COLORS, OBJECT_CLASS_COLORS,
    get_T_cam_from_master, get_K_from_calib,
    box_8_corners, project_to_image, load_kitti_frame, load_radar_points,
    is_radar_only_object,
)

# ─── Theme constants ─────────────────────────────────────────────────────────
PANEL_BG  = "#0b1525"
BORDER    = "#152436"
TEXT_DIM  = "#7793ae"
CYAN      = "#00e5ff"
RADAR_RED = "#ff0038"
RADAR_GREEN = "#39ff57"
ROAD_EDGE = "#4a4f55"
LANE_LINE = "#3d454d"
FONT_MON  = "Share Tech Mono, monospace"
FONT_ORB  = "Orbitron, monospace"

RADAR_FORWARD_RANGE_M = 90.0
RADAR_PLANE_BACK_M = 0.0
RADAR_LATERAL_RANGE_M = 80.0
RADAR_DISPLAY_LATERAL_RANGE_M = 80.0
RADAR_ROAD_CORE_HALF_WIDTH_M = 12.0
RADAR_MIN_VIEW_HALF_WIDTH_M = 20.0
RADAR_COL_RCS = 3
RADAR_COL_VR = 4
RADAR_COL_VR_COMP = 5
RADAR_OBJECT_VEL_THRESHOLD_MS = 0.55
RADAR_OBJECT_COMP_THRESHOLD_MS = 0.55
RADAR_OBJECT_RCS_PERCENTILE = 85.0
RADAR_COMP_ONLY_RAW_EPS_MS = 0.55
RADAR_COMP_ONLY_COMP_MIN_MS = 2.0
RADAR_CLASS_COLORS = OBJECT_CLASS_COLORS
# (min L,W,H), (max L,W,H) — centred on radar_annotator_v3 DEFAULT_BOX_SPECS so
# proportions match the annotator. Pedestrians/riders are tall-and-thin (upright
# people), which makes them read as true 3D volumes rather than flat cubes.
RADAR_BOX_DIM_LIMITS = {
    "Car": ((3.8, 1.6, 1.35), (5.2, 2.1, 1.85)),
    "Truck": ((5.5, 2.2, 2.4), (12.0, 3.0, 3.6)),
    "Van": ((4.3, 1.8, 1.7), (6.5, 2.4, 2.5)),
    "Bus": ((8.0, 2.4, 2.8), (14.0, 3.2, 3.8)),
    "Other vehicle": ((2.6, 1.2, 1.4), (5.5, 2.3, 2.4)),
    "Motorcycle": ((1.8, 0.6, 1.25), (2.6, 1.0, 1.7)),
    "Motorbike": ((1.8, 0.6, 1.25), (2.6, 1.0, 1.7)),
    "E-Scooter / Moped": ((1.6, 0.6, 1.25), (2.3, 0.95, 1.65)),
    "Cyclist": ((1.5, 0.5, 1.6), (2.2, 0.9, 1.95)),
    "Bicycle": ((1.5, 0.5, 1.1), (2.2, 0.9, 1.4)),
    "Rider": ((0.5, 0.5, 1.6), (0.95, 0.95, 1.95)),
    "Pedestrian": ((0.45, 0.45, 1.6), (0.9, 0.9, 1.95)),
    "Animal": ((0.8, 0.4, 0.7), (2.6, 1.1, 1.7)),
}
RADAR_DEPTH_BINS_M = (
    (0.0, 10.0, 1.2, 4),
    (10.0, 20.0, 1.4, 3),
    (20.0, 40.0, 1.8, 2),
    (40.0, 60.0, 2.4, 2),
    (60.0, 80.0, 3.0, 2),
    (80.0, 90.0, 3.4, 2),
)
RADAR_Z_RANGE_M = (-0.05, 2.0)
# Lateral half-width (m) of the 3D road corridor — full sensor field of view.
RADAR_3D_LATERAL_HALF_M = 80.0
# center/eye are in Plotly's normalised scene coordinates (box centred near
# origin), NOT metres. The aspect ratio keeps the x/y/z per-metre scales close
# so 3D boxes keep realistic car proportions; perspective — not axis stretching
# — makes the road fill the pane from the origin (bottom) up to the horizon.
# The camera is yawed to the right so the road head points toward the upper
# right of the pane (matching the heading arrow), origin visible at the bottom.
RADAR_3D_ASPECT_RATIO = dict(x=1.9, y=1.18, z=0.030)
RADAR_3D_CAMERA = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=-0.16, y=0.0, z=-0.20),
    eye=dict(x=0.28, y=-0.85, z=0.36),
    projection=dict(type="perspective"),
)
RADAR_BOX_ASSOC_MARGIN_M = 0.20
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
BOX_FRONT_FACE = (1, 2, 6, 5)
BOX_TOP_FACE = (4, 5, 6, 7)
BOX_FACES_FILL = (
    (0, 3, 7, 4),     # back
    (0, 1, 2, 3),     # bottom
    BOX_TOP_FACE,
    (0, 1, 5, 4),     # right side
    (3, 2, 6, 7),     # left side
    BOX_FRONT_FACE,
)
BOX_FRONT_EDGE_SET = frozenset(
    frozenset(edge) for edge in ((1, 2), (2, 6), (6, 5), (5, 1))
)

# ─── Utility ─────────────────────────────────────────────────────────────────
def rgba(hex_col: str, alpha: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def brighten(hex_col: str, amount: int = 45) -> str:
    h = hex_col.lstrip("#")
    r = min(255, int(h[0:2], 16) + amount)
    g = min(255, int(h[2:4], 16) + amount)
    b = min(255, int(h[4:6], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _base_layout(**overrides) -> dict:
    """Shared dark layout settings."""
    base = dict(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family=FONT_MON, color=TEXT_DIM),
        margin=dict(l=6, r=6, t=6, b=6),
        showlegend=False,
        dragmode=False,
        hovermode="closest",
    )
    base.update(overrides)
    return base


def _visible_radar_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 7)
    mask = (
        (points[:, 0] >= 0.0) &
        (points[:, 0] <= RADAR_FORWARD_RANGE_M) &
        (np.abs(points[:, 1]) <= RADAR_LATERAL_RANGE_M) &
        np.all(np.isfinite(points[:, :3]), axis=1)
    )
    return points[mask]


# Cap radar markers sent to Plotly — full clouds make WebGL/browser redraws stutter.


def _adaptive_threshold(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, percentile))


def _local_density_mask(points_xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    n = points_xyz.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.array([min_neighbors <= 1], dtype=bool)

    radius2 = float(radius) * float(radius)
    keep = np.zeros(n, dtype=bool)
    chunk = 512
    for start in range(0, n, chunk):
        block = points_xyz[start:start + chunk]
        diff = block[:, None, :] - points_xyz[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        counts = np.count_nonzero(dist2 <= radius2, axis=1)
        keep[start:start + chunk] = counts >= min_neighbors
    return keep


def _radar_column(points: np.ndarray, column: int) -> np.ndarray:
    if not points.size or points.shape[1] <= column:
        return np.empty((0,), dtype=np.float64)
    values = points[:, column].astype(np.float64)
    return np.where(np.isfinite(values), values, 0.0)


def _raw_velocity_column(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return _radar_column(points, RADAR_COL_VR) if points.shape[1] > RADAR_COL_VR else np.zeros(points.shape[0])


def _comp_velocity_column(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if points.shape[1] > RADAR_COL_VR_COMP:
        comp = points[:, RADAR_COL_VR_COMP].astype(np.float64)
        raw = _raw_velocity_column(points)
        return np.where(np.isfinite(comp), comp, raw)
    return _raw_velocity_column(points)


def _compensation_only_mask(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points if points is not None else np.zeros((0, 7)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] <= RADAR_COL_VR_COMP:
        return np.zeros((pts.shape[0] if pts.ndim == 2 else 0,), dtype=bool)
    raw = _raw_velocity_column(pts)
    comp = _comp_velocity_column(pts)
    return (
        np.isfinite(raw) &
        np.isfinite(comp) &
        (np.abs(raw) <= RADAR_COMP_ONLY_RAW_EPS_MS) &
        (np.abs(comp) >= RADAR_COMP_ONLY_COMP_MIN_MS)
    )


def _valid_doppler_mask(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points if points is not None else np.zeros((0, 7)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((pts.shape[0] if pts.ndim == 2 else 0,), dtype=bool)
    raw = _raw_velocity_column(pts)
    return np.isfinite(raw) & (np.abs(raw) >= RADAR_OBJECT_VEL_THRESHOLD_MS)


def _radar_rcs_mask(pts: np.ndarray, finite_xyz: np.ndarray, rcs: np.ndarray) -> np.ndarray:
    global_threshold = _adaptive_threshold(rcs[finite_xyz], RADAR_OBJECT_RCS_PERCENTILE)
    keep = np.zeros(pts.shape[0], dtype=bool)
    covered = np.zeros(pts.shape[0], dtype=bool)
    depth = pts[:, 0]
    for depth_min, depth_max, _, _ in RADAR_DEPTH_BINS_M:
        in_bin = finite_xyz & (depth >= depth_min) & (depth < depth_max)
        if not in_bin.any():
            continue
        covered |= in_bin
        bin_threshold = _adaptive_threshold(rcs[in_bin], RADAR_OBJECT_RCS_PERCENTILE)
        threshold = min(global_threshold, bin_threshold)
        keep |= in_bin & np.isfinite(rcs) & (rcs >= threshold)

    outside_bins = finite_xyz & ~covered
    if outside_bins.any():
        keep |= outside_bins & np.isfinite(rcs) & (rcs >= global_threshold)
    return keep


def _radar_density_mask(pts: np.ndarray, pre: np.ndarray) -> np.ndarray:
    mask = np.zeros(pts.shape[0], dtype=bool)
    depth = pts[:, 0]
    for depth_min, depth_max, radius, min_neighbors in RADAR_DEPTH_BINS_M:
        active = np.where(pre & (depth >= depth_min) & (depth < depth_max))[0]
        if active.size == 0:
            continue
        keep = _local_density_mask(pts[active, :3], radius=radius, min_neighbors=int(min_neighbors))
        mask[active] = keep
    return mask


def _object_vel_mask(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points if points is not None else np.zeros((0, 7)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] <= RADAR_COL_RCS:
        return np.zeros((pts.shape[0] if pts.ndim == 2 else 0,), dtype=bool)

    finite_xyz = np.all(np.isfinite(pts[:, :3]), axis=1)
    rcs = pts[:, RADAR_COL_RCS]
    raw_velocity = _raw_velocity_column(pts)
    compensated_velocity = _comp_velocity_column(pts)
    same_direction = np.sign(raw_velocity) == np.sign(compensated_velocity)
    vel_keep = (
        np.isfinite(raw_velocity) &
        np.isfinite(compensated_velocity) &
        (np.abs(raw_velocity) >= RADAR_OBJECT_VEL_THRESHOLD_MS) &
        (np.abs(compensated_velocity) >= RADAR_OBJECT_COMP_THRESHOLD_MS) &
        same_direction
    )
    rcs_keep = _radar_rcs_mask(pts, finite_xyz, rcs)
    pre = finite_xyz & vel_keep & rcs_keep & ~_compensation_only_mask(pts)
    if not pre.any():
        return pre
    return pre & _radar_density_mask(pts, pre)


# ─── BEV Map ─────────────────────────────────────────────────────────────────
def _object_point_mask(points: np.ndarray, objects: Optional[List[TrackedObject]]) -> np.ndarray:
    """Return radar points whose lateral/depth footprint falls inside a tracked 3D box."""
    pts = np.asarray(points if points is not None else np.zeros((0, 7)), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or not objects:
        return np.zeros((pts.shape[0] if pts.ndim == 2 else 0,), dtype=bool)

    keep = np.zeros(pts.shape[0], dtype=bool)
    for obj in objects:
        if obj.bev_y < -2 or obj.bev_y > RADAR_FORWARD_RANGE_M + 2:
            continue
        if abs(obj.bev_x) > RADAR_LATERAL_RANGE_M + 2:
            continue

        length = max(float(obj.box_l), 0.2)
        width = max(float(obj.box_w), 0.2)
        yaw = float(obj.box_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        rel_forward = pts[:, 0] - float(obj.bev_y)
        rel_lateral = pts[:, 1] - float(obj.bev_x)
        local_length = c * rel_forward + s * rel_lateral
        local_width = -s * rel_forward + c * rel_lateral
        margin = RADAR_BOX_ASSOC_MARGIN_M
        keep |= (
            (np.abs(local_length) <= length * 0.5 + margin)
            & (np.abs(local_width) <= width * 0.5 + margin)
        )
    return keep


def _radar_box_dimensions(obj: TrackedObject) -> tuple[float, float, float]:
    length = max(float(obj.box_l), 0.2)
    width = max(float(obj.box_w), 0.2)
    height = max(float(obj.box_h), 0.2)
    limits = RADAR_BOX_DIM_LIMITS.get(obj.cls)
    if limits is not None:
        min_dims, max_dims = limits
        length = min(max(length, min_dims[0]), max_dims[0])
        width = min(max(width, min_dims[1]), max_dims[1])
        height = min(max(height, min_dims[2]), max_dims[2])
    return length, width, height


def _radar_object_color(obj: TrackedObject, index: int) -> str:
    return RADAR_CLASS_COLORS.get(obj.cls, "#64748b")


def _box_corners_3d(obj: TrackedObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length, width, height = _radar_box_dimensions(obj)
    center_master = np.array([float(obj.bev_y), float(obj.bev_x), height * 0.5])
    corners = box_8_corners(center_master, length, width, height, float(obj.box_yaw))
    return corners[:, 1], corners[:, 0], corners[:, 2]


def _edge_segments_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in BOX_EDGES:
        xs.extend([float(x[a]), float(x[b]), None])
        ys.extend([float(y[a]), float(y[b]), None])
        zs.extend([float(z[a]), float(z[b]), None])
    return xs, ys, zs


def _vertical_edge_segments_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in ((0, 4), (1, 5), (2, 6), (3, 7)):
        xs.extend([float(x[a]), float(x[b]), None])
        ys.extend([float(y[a]), float(y[b]), None])
        zs.extend([float(z[a]), float(z[b]), None])
    return xs, ys, zs


def _radar_plot_z(z_values: np.ndarray) -> np.ndarray:
    z = np.asarray(z_values, dtype=np.float64)
    return np.clip(z, RADAR_Z_RANGE_M[0] + 0.04, RADAR_Z_RANGE_M[1] - 0.04)


def _radar_scene_x_range(pts: np.ndarray, objects: Optional[List[TrackedObject]]) -> list[float]:
    values: list[float] = []
    if pts.size:
        finite = pts[np.isfinite(pts[:, 1])]
        if finite.size:
            values.extend(float(v) for v in finite[:, 1])
    if objects:
        for obj in objects:
            if obj.bev_y < -2 or obj.bev_y > RADAR_FORWARD_RANGE_M + 5:
                continue
            x, _, _ = _box_corners_3d(obj)
            values.extend(float(v) for v in x if np.isfinite(v))

    values.extend([-RADAR_ROAD_CORE_HALF_WIDTH_M, RADAR_ROAD_CORE_HALF_WIDTH_M])
    x_min = max(-RADAR_DISPLAY_LATERAL_RANGE_M, min(values) - 6.0)
    x_max = min(RADAR_DISPLAY_LATERAL_RANGE_M, max(values) + 6.0)
    center = (x_min + x_max) * 0.5
    half = max((x_max - x_min) * 0.5, RADAR_MIN_VIEW_HALF_WIDTH_M)
    x_min = max(-RADAR_DISPLAY_LATERAL_RANGE_M, center - half)
    x_max = min(RADAR_DISPLAY_LATERAL_RANGE_M, center + half)
    return [float(x_min), float(x_max)]


def _project_world_points(points_master: np.ndarray, T_cam_from_master: np.ndarray,
                          K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_master, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    homog = np.column_stack((pts[:, :3], np.ones(pts.shape[0], dtype=np.float64)))
    cam = (T_cam_from_master @ homog.T).T[:, :3]
    depths = cam[:, 2]
    uv = np.full((pts.shape[0], 2), -1.0, dtype=np.float64)
    front = depths > 0.05
    if np.any(front):
        pix = (K @ cam[front].T).T
        uv[front, 0] = pix[:, 0] / np.maximum(pix[:, 2], 1e-9)
        uv[front, 1] = pix[:, 1] / np.maximum(pix[:, 2], 1e-9)
    return uv, depths


def _add_road_line_3d(fig: go.Figure, lateral0: float, depth0: float, lateral1: float,
                      depth1: float, *, color: str, width: float, z: float = 0.0) -> None:
    samples = max(2, int(abs(depth1 - depth0) / 5.0) + 2)
    lateral = np.linspace(lateral0, lateral1, samples)
    depth = np.linspace(depth0, depth1, samples)
    fig.add_trace(go.Scatter3d(
        x=lateral,
        y=depth,
        z=np.full(samples, z),
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
    ))


def _append_road_line_3d(xs: list, ys: list, zs: list,
                         lateral0: float, depth0: float, lateral1: float,
                         depth1: float, z: float = 0.0) -> None:
    samples = max(2, int(abs(depth1 - depth0) / 5.0) + 2)
    lateral = np.linspace(lateral0, lateral1, samples)
    depth = np.linspace(depth0, depth1, samples)
    xs.extend(float(v) for v in lateral)
    ys.extend(float(v) for v in depth)
    zs.extend([float(z)] * samples)
    xs.append(None)
    ys.append(None)
    zs.append(None)


def _add_road_surface_3d(fig: go.Figure, x_range: list[float]) -> None:
    road = RADAR_ROAD_CORE_HALF_WIDTH_M
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min = -RADAR_PLANE_BACK_M
    bands = []
    if x_min < -road:
        bands.append((x_min, -road, "#020304", 0.94))
    bands.append((max(x_min, -road), min(x_max, road), "#070b0f", 1.0))
    if x_max > road:
        bands.append((road, x_max, "#020304", 0.94))
    for left, right, color, opacity in bands:
        if right <= left:
            continue
        fig.add_trace(go.Mesh3d(
            x=[left, right, right, left],
            y=[y_min, y_min, RADAR_FORWARD_RANGE_M, RADAR_FORWARD_RANGE_M],
            z=[0.0, 0.0, 0.0, 0.0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=color,
            opacity=opacity,
            hoverinfo="skip",
            showscale=False,
            showlegend=False,
        ))


def _add_depth_reference_3d(fig: go.Figure, x_range: list[float]) -> None:
    road = RADAR_ROAD_CORE_HALF_WIDTH_M
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min = -RADAR_PLANE_BACK_M

    def add_group(xs: list, ys: list, zs: list, color: str, width: float) -> None:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        ))

    edge_xs, edge_ys, edge_zs = [], [], []
    guide_xs, guide_ys, guide_zs = [], [], []
    lane_xs, lane_ys, lane_zs = [], [], []
    grid_xs, grid_ys, grid_zs = [], [], []
    center_xs, center_ys, center_zs = [], [], []

    _append_road_line_3d(edge_xs, edge_ys, edge_zs, x_min, y_min, x_min, RADAR_FORWARD_RANGE_M)
    _append_road_line_3d(edge_xs, edge_ys, edge_zs, x_max, y_min, x_max, RADAR_FORWARD_RANGE_M)
    _append_road_line_3d(guide_xs, guide_ys, guide_zs, -road, y_min, -road, RADAR_FORWARD_RANGE_M)
    _append_road_line_3d(guide_xs, guide_ys, guide_zs, road, y_min, road, RADAR_FORWARD_RANGE_M)
    _append_road_line_3d(center_xs, center_ys, center_zs, 0, y_min, 0, RADAR_FORWARD_RANGE_M)

    for lateral in (-6.0, -3.0, 3.0, 6.0):
        _append_road_line_3d(guide_xs, guide_ys, guide_zs, lateral, y_min, lateral, RADAR_FORWARD_RANGE_M)

    for lateral in (-1.25, 1.25):
        for depth0 in range(6, int(RADAR_FORWARD_RANGE_M), 12):
            _append_road_line_3d(
                lane_xs, lane_ys, lane_zs, lateral, depth0, lateral,
                min(depth0 + 6, RADAR_FORWARD_RANGE_M)
            )

    for depth in [0, *range(10, int(RADAR_FORWARD_RANGE_M) + 1, 10)]:
        major = depth % 20 == 0 and depth < int(RADAR_FORWARD_RANGE_M)
        _append_road_line_3d(grid_xs, grid_ys, grid_zs, x_min, depth, x_max, depth)
        if major:
            for lateral in (-road, road):
                grid_xs.extend([lateral, lateral, None])
                grid_ys.extend([depth, depth, None])
                grid_zs.extend([0.0, 1.2, None])

    add_group(edge_xs, edge_ys, edge_zs, ROAD_EDGE, 7)
    add_group(guide_xs, guide_ys, guide_zs, "#59626b", 5)
    add_group(lane_xs, lane_ys, lane_zs, LANE_LINE, 5)
    add_group(grid_xs, grid_ys, grid_zs, "#294052", 3)
    add_group(center_xs, center_ys, center_zs, RADAR_GREEN, 6)


def _add_radar_box_overlays_3d(fig: go.Figure, objects: Optional[List[TrackedObject]]) -> None:
    if not objects:
        return

    base_i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    base_j = [1, 2, 6, 7, 1, 5, 2, 6, 3, 7, 0, 4]
    base_k = [2, 3, 5, 6, 5, 4, 6, 5, 7, 6, 4, 7]
    groups: dict[str, dict] = {}

    for index, obj in enumerate(objects):
        if obj.bev_y < -2 or obj.bev_y > RADAR_FORWARD_RANGE_M + 5:
            continue
        if abs(obj.bev_x) > RADAR_3D_LATERAL_HALF_M + 3:
            continue

        col = _radar_object_color(obj, index)
        front_col = brighten(col, 70)
        group = groups.setdefault(obj.cls, {
            "color": col,
            "front": front_col,
            "mesh_x": [], "mesh_y": [], "mesh_z": [], "mesh_hover": [],
            "mesh_i": [], "mesh_j": [], "mesh_k": [],
            "edge_x": [], "edge_y": [], "edge_z": [],
            "bright_x": [], "bright_y": [], "bright_z": [],
            "heading_x": [], "heading_y": [], "heading_z": [],
            "label_x": [], "label_y": [], "label_z": [],
            "labels": [], "hovers": [],
        })

        x, y, z = _box_corners_3d(obj)
        z = _radar_plot_z(z)
        length, width, height = _radar_box_dimensions(obj)
        ttc_txt = f"{obj.ttc:.1f} s" if math.isfinite(obj.ttc) else "—"
        hover = (
            f"<b>ID {obj.id} · {obj.cls}</b>  <span style='color:{RISK_COLORS.get(obj.risk, '#8ab4cc')}'>[{obj.risk}]</span><br>"
            f"distance = {obj.dist:.1f} m  ·  depth = {obj.bev_y:.1f} m  ·  lateral = {obj.bev_x:+.1f} m<br>"
            f"rel. speed = {obj.rel_vel:+.1f} m/s  ·  TTC = {ttc_txt}  ·  req. decel = {obj.req_decel:.1f} m/s²<br>"
            f"size L×W×H = {length:.2f} × {width:.2f} × {height:.2f} m  ·  yaw = {math.degrees(obj.box_yaw):.0f}°<br>"
            f"source = {obj.source}  ·  confidence = {obj.confidence*100:.0f}%"
        )

        offset = len(group["mesh_x"])
        group["mesh_x"].extend(float(v) for v in x)
        group["mesh_y"].extend(float(v) for v in y)
        group["mesh_z"].extend(float(v) for v in z)
        group["mesh_hover"].extend([hover] * len(x))
        group["mesh_i"].extend(offset + i for i in base_i)
        group["mesh_j"].extend(offset + j for j in base_j)
        group["mesh_k"].extend(offset + k for k in base_k)

        edge_x, edge_y, edge_z = _edge_segments_3d(x, y, z)
        group["edge_x"].extend(edge_x)
        group["edge_y"].extend(edge_y)
        group["edge_z"].extend(edge_z)

        vertical_x, vertical_y, vertical_z = _vertical_edge_segments_3d(x, y, z)
        front_edge_x, front_edge_y, front_edge_z = [], [], []
        for a, b in ((1, 2), (2, 6), (6, 5), (5, 1)):
            front_edge_x.extend([float(x[a]), float(x[b]), None])
            front_edge_y.extend([float(y[a]), float(y[b]), None])
            front_edge_z.extend([float(z[a]), float(z[b]), None])
        group["bright_x"].extend(front_edge_x + vertical_x)
        group["bright_y"].extend(front_edge_y + vertical_y)
        group["bright_z"].extend(front_edge_z + vertical_z)

        yaw = float(obj.box_yaw)
        front_lateral = float(obj.bev_x) + math.sin(yaw) * length * 0.58
        front_depth = float(obj.bev_y) + math.cos(yaw) * length * 0.58
        heading_z = 0.08
        group["heading_x"].extend([float(obj.bev_x), front_lateral, None])
        group["heading_y"].extend([float(obj.bev_y), front_depth, None])
        group["heading_z"].extend([heading_z, heading_z, None])

        lz = min(height + 0.25, RADAR_Z_RANGE_M[1] - 0.08)
        group["label_x"].append(float(obj.bev_x))
        group["label_y"].append(float(obj.bev_y))
        group["label_z"].append(lz)
        group["labels"].append(f"<b>{obj.id} · {obj.cls}</b>")
        group["hovers"].append(hover)

    for group in groups.values():
        if not group["mesh_x"]:
            continue
        fig.add_trace(go.Mesh3d(
            x=group["mesh_x"], y=group["mesh_y"], z=group["mesh_z"],
            i=group["mesh_i"], j=group["mesh_j"], k=group["mesh_k"],
            color=group["color"],
            opacity=0.28,
            flatshading=True,
            hovertext=group["mesh_hover"],
            hoverinfo="text",
            showscale=False,
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=group["edge_x"], y=group["edge_y"], z=group["edge_z"],
            mode="lines",
            line=dict(color="rgba(1,5,12,0.94)", width=9),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=group["edge_x"], y=group["edge_y"], z=group["edge_z"],
            mode="lines",
            line=dict(color=group["color"], width=4),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=group["bright_x"], y=group["bright_y"], z=group["bright_z"],
            mode="lines",
            line=dict(color=group["front"], width=6),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=group["heading_x"], y=group["heading_y"], z=group["heading_z"],
            mode="lines",
            line=dict(color=group["front"], width=5),
            hoverinfo="skip",
            showlegend=False,
        ))
        # No persistent text label — full details appear on hover only. A small
        # marker at the box top acts as an extra hover anchor alongside the mesh.
        fig.add_trace(go.Scatter3d(
            x=group["label_x"],
            y=group["label_y"],
            z=group["label_z"],
            mode="markers",
            marker=dict(size=4.5, color=group["front"], opacity=0.9),
            hovertext=group["hovers"],
            hoverinfo="text",
            showlegend=False,
        ))


def _build_radar_3d_figure(frame_id: str, objects: Optional[List[TrackedObject]] = None,
                           n_intervals: int = 0) -> go.Figure:
    pts = _visible_radar_points(load_radar_points(frame_id))
    fig = go.Figure()
    x_range = [-RADAR_3D_LATERAL_HALF_M, RADAR_3D_LATERAL_HALF_M]

    _add_road_surface_3d(fig, x_range)
    _add_depth_reference_3d(fig, x_range)

    if pts.size:
        object_mask = _object_vel_mask(pts)

        def add_cloud(name: str, cloud: np.ndarray, color: Optional[str], size: float,
                      opacity: float, outline: bool = False,
                      depth_rgb: bool = False) -> None:
            if cloud.size == 0:
                return
            rcs = _radar_column(cloud, RADAR_COL_RCS)
            raw_velocity = _raw_velocity_column(cloud)
            compensated_velocity = _comp_velocity_column(cloud)
            near_scale = np.clip((RADAR_FORWARD_RANGE_M - cloud[:, 0]) / RADAR_FORWARD_RANGE_M, 0.0, 1.0)
            marker_size = np.clip(size + near_scale * 1.6, size, size + 1.6)
            if depth_rgb:
                marker = dict(
                    size=marker_size,
                    color=cloud[:, 0],
                    cmin=0.0,
                    cmax=RADAR_FORWARD_RANGE_M,
                    colorscale=[
                        [0.0, "rgb(30, 185, 255)"],
                        [0.30, "rgb(0, 230, 255)"],
                        [0.56, "rgb(0, 230, 120)"],
                        [0.78, "rgb(255, 210, 0)"],
                        [1.0, "rgb(255, 112, 44)"],
                    ],
                    opacity=opacity,
                    showscale=False,
                )
            else:
                marker = dict(size=marker_size, color=color, opacity=opacity)
            if outline:
                marker["line"] = dict(color="#ffd3dc", width=1.2)
            fig.add_trace(go.Scatter3d(
                x=cloud[:, 1],
                y=cloud[:, 0],
                z=np.full(cloud.shape[0], 0.035, dtype=np.float64),
                mode="markers",
                name=name,
                marker=marker,
                hovertemplate=(
                    "lat=%{customdata[3]:.1f} m<br>depth=%{customdata[4]:.1f} m"
                    "<br>z=%{customdata[5]:.1f} m"
                    "<br>RCS=%{customdata[0]:.1f}"
                    "<br>raw vel=%{customdata[1]:.2f} m/s"
                    "<br>comp vel=%{customdata[2]:.2f} m/s"
                    "<extra></extra>"
                ),
                customdata=np.column_stack((
                    rcs, raw_velocity, compensated_velocity,
                    cloud[:, 1], cloud[:, 0], cloud[:, 2],
                )),
                showlegend=False,
            ))

        add_cloud("EFEAR radar cloud", pts, CYAN, 1.8, 0.74)
        add_cloud("moving object radar points", pts[object_mask], RADAR_RED, 4.8, 0.98, outline=True)
    else:
        fig.add_annotation(
            x=0.5, y=0.5, text=f"NO RADAR POINTS - FRAME {frame_id}",
            showarrow=False, font=dict(size=11, color=TEXT_DIM, family=FONT_MON),
        )

    _add_radar_box_overlays_3d(fig, objects)
    fig.update_layout(**_base_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            bgcolor="black",
            xaxis=dict(title="", range=x_range,
                       visible=False, showbackground=False, showgrid=False,
                       zeroline=False),
            yaxis=dict(title="", range=[0.0, RADAR_FORWARD_RANGE_M],
                       visible=False, showbackground=False, showgrid=False,
                       zeroline=False),
            zaxis=dict(title="", range=list(RADAR_Z_RANGE_M),
                       visible=False, showbackground=False, showgrid=False,
                       zeroline=False),
            aspectmode="manual",
            aspectratio=RADAR_3D_ASPECT_RATIO,
            camera=RADAR_3D_CAMERA,
        ),
        scene_dragmode=False,
        dragmode=False,
        hovermode="closest",
        uirevision="dashboard-3d-radar-view-v5",
    ))
    return fig


# ─── Camera View ──────────────────────────────────────────────────────────────
def build_bev_figure(frame_id: str, objects: Optional[List[TrackedObject]] = None,
                     n_intervals: int = 0) -> go.Figure:
    """EFEAR-style 3D radar road view for the dashboard BEV panel."""
    return _build_radar_3d_figure(frame_id, objects, n_intervals)


def build_camera_figure(
    objects: List[TrackedObject],
    frame_id: str = "0000",
    *,
    embed_image: bool = False,
) -> go.Figure:
    """
    Camera view with actual KITTI image and detection overlays:
    - Load actual KITTI image as background
    - Overlay detection bounding boxes with corner markers, labels, stats

    When ``embed_image`` is True, the RGB frame is inlined as a data-URI so
    static export HTML / Playwright screenshots work without the Flask server.
    """
    from data import load_kitti_frame, get_frame_image_data_uri

    frame_data = load_kitti_frame(frame_id)
    image_path = frame_data['image_path']
    if os.path.exists(image_path):
        image_source = (
            get_frame_image_data_uri(frame_id)
            if embed_image
            else f"/frame-image/{frame_id}"
        )
    else:
        image_source = None

    fig = go.Figure()

    # Preserve image pixel geometry so detection boxes stay registered to objects.
    width, height = frame_data.get('image_size', (1, 1))
    fig.update_layout(**_base_layout(
        paper_bgcolor="#060d18",
        plot_bgcolor="#060d18",
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        # Lock the pixel aspect ratio so the frame is never stretched. The y-axis
        # is scale-anchored to x at 1:1; putting `constrain="domain"` on the
        # Y axis (not X) makes the X domain stay full-width while the Y domain
        # shrinks to keep pixels square — so a wide frame fills the panel
        # LEFT-TO-RIGHT and only letterboxes top/bottom (never pillar-boxes).
        # Overlays stay registered because they share these axes.
        xaxis=dict(range=[0, width], showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True),
        yaxis=dict(range=[height, 0], showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True, scaleanchor="x", scaleratio=1, constrain="domain"),
    ))

    # Add the actual image if available
    if image_source:
        fig.add_layout_image(
            dict(
                source=image_source,
                xref="x",
                yref="y",
                x=0,
                y=0,          # y=0 is top with range=[height,0]
                xanchor="left",
                yanchor="top",
                sizex=width,
                sizey=height,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
    else:
        # Fallback: simulated road surface
        fig.add_trace(go.Scatter(
            x=[0, 640, 467, 173, 0], y=[0, 0, 230, 230, 0],
            fill="toself", fillcolor="rgba(10,24,7,0.85)",
            line=dict(color="rgba(255,255,255,0.08)", width=1),
            hoverinfo="skip",
        ))

    # ── Calibration for 3-D wireframe
    calib          = frame_data.get('calib', {})
    K              = get_K_from_calib(calib, (width, height))  # 3×3 or None
    T_cam_from_mst = get_T_cam_from_master(calib)      # 4×4

    _WIRE_EDGES = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]

    shapes = []
    annotations = []

    # ── Object overlays
    CS = 15   # corner stripe length (px)
    for o in objects:
        if is_radar_only_object(o.occluded):
            continue
        if o.cam_cx < 0 or o.cam_w < 0 or o.cam_h < 0:
            continue
        col     = OBJECT_CLASS_COLORS.get(o.cls, "#64748b")

        x0 = o.cam_cx - o.cam_w / 2
        x1 = o.cam_cx + o.cam_w / 2
        y0 = o.cam_cy - o.cam_h / 2   # top in image coords
        y1 = o.cam_cy + o.cam_h / 2   # bottom

        # ── 3D wireframe (12 edges) projected via calibration
        wire_drawn = False
        projected_bbox = None
        if K is not None:
            center_master = np.array([o.bev_y, o.bev_x, o.box_z])
            corners = box_8_corners(center_master, o.box_l, o.box_w, o.box_h, o.box_yaw)
            uvs, depths = project_to_image(corners, T_cam_from_mst, K)
            front = depths > 0.5
            if np.any(front):
                visible_uv = uvs[front]
                projected_bbox = (
                    float(np.min(visible_uv[:, 0])),
                    float(np.min(visible_uv[:, 1])),
                    float(np.max(visible_uv[:, 0])),
                    float(np.max(visible_uv[:, 1])),
                )

            faces_ranked = []
            for face in BOX_FACES_FILL:
                if np.all(depths[list(face)] > 0.5):
                    faces_ranked.append((float(np.mean(depths[list(face)])), face))
            faces_ranked.sort(key=lambda item: -item[0])
            for _mean_depth, face in faces_ranked:
                face_uv = uvs[list(face)]
                closed_x = [float(v) for v in face_uv[:, 0]] + [float(face_uv[0, 0])]
                closed_y = [float(v) for v in face_uv[:, 1]] + [float(face_uv[0, 1])]
                is_front_face = tuple(face) == BOX_FRONT_FACE
                fig.add_trace(go.Scatter(
                    x=closed_x,
                    y=closed_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=rgba(
                        brighten(col, 55) if is_front_face else col,
                        0.26 if is_front_face else 0.08,
                    ),
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                wire_drawn = True
            for i_e, j_e in _WIRE_EDGES:
                if depths[i_e] > 0.5 and depths[j_e] > 0.5:
                    is_front_edge = frozenset((i_e, j_e)) in BOX_FRONT_EDGE_SET
                    line_w = 3.1 if is_front_edge else 2.0
                    shapes.append(dict(type="line",
                        x0=float(uvs[i_e, 0]), y0=float(uvs[i_e, 1]),
                        x1=float(uvs[j_e, 0]), y1=float(uvs[j_e, 1]),
                        line=dict(color="rgba(2,6,23,0.86)", width=line_w + 2.2)))
                    shapes.append(dict(type="line",
                        x0=float(uvs[i_e, 0]), y0=float(uvs[i_e, 1]),
                        x1=float(uvs[j_e, 0]), y1=float(uvs[j_e, 1]),
                        line=dict(color=brighten(col, 40) if is_front_edge else col,
                                  width=line_w)))
                    wire_drawn = True

            center_uv, center_depth = _project_world_points(center_master, T_cam_from_mst, K)
            tip_master = center_master + np.array([
                math.cos(o.box_yaw) * min(max(o.box_l * 0.34, 0.7), 1.5),
                math.sin(o.box_yaw) * min(max(o.box_l * 0.34, 0.7), 1.5),
                0.0,
            ])
            tip_uv, tip_depth = _project_world_points(tip_master, T_cam_from_mst, K)
            if center_depth[0] > 0.5 and tip_depth[0] > 0.5:
                shapes.append(dict(type="line",
                    x0=float(center_uv[0, 0]), y0=float(center_uv[0, 1]),
                    x1=float(tip_uv[0, 0]), y1=float(tip_uv[0, 1]),
                    line=dict(color="rgba(2,6,23,0.88)", width=5.0)))
                shapes.append(dict(type="line",
                    x0=float(center_uv[0, 0]), y0=float(center_uv[0, 1]),
                    x1=float(tip_uv[0, 0]), y1=float(tip_uv[0, 1]),
                    line=dict(color="#ffffff", width=2.4)))

        # ── 2D fallback box (always drawn — provides the filled background rect)
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=rgba(col, 0.08 if not wire_drawn else 0.0),
            line=dict(color=col if not wire_drawn else "rgba(0,0,0,0)", width=2)))

        # Tactical corner markers
        for cpx, cpy, sx, sy in [(x0,y0,1,1),(x1,y0,-1,1),(x1,y1,-1,-1),(x0,y1,1,-1)]:
            shapes.append(dict(type="line", x0=cpx, y0=cpy, x1=cpx+sx*CS, y1=cpy,
                line=dict(color=col if not wire_drawn else "rgba(0,0,0,0)", width=3)))
            shapes.append(dict(type="line", x0=cpx, y0=cpy, x1=cpx, y1=cpy+sy*CS,
                line=dict(color=col if not wire_drawn else "rgba(0,0,0,0)", width=3)))

        if projected_bbox is not None:
            lx0, ly0, lx1, ly1 = projected_bbox
        else:
            lx0, ly0, lx1, ly1 = x0, y0, x1, y1
        ttc_str = f"{o.ttc:.1f}" if math.isfinite(o.ttc) else "∞"

        # No persistent label — full details are shown on hover only. An invisible
        # filled rectangle over the box acts as the hover target for that object.
        hover = (
            f"<b>ID {o.id} · {o.cls.upper()}</b>  "
            f"<span style='color:{RISK_COLORS.get(o.risk, '#8ab4cc')}'>[{o.risk}]</span><br>"
            f"distance = {o.dist:.1f} m  ·  rel. speed = {o.rel_vel:+.1f} m/s<br>"
            f"TTC = {ttc_str} s  ·  req. decel = {o.req_decel:.1f} m/s²<br>"
            f"heading = {o.heading}  ·  source = {o.source}  ·  conf = {o.confidence*100:.0f}%"
        )
        fig.add_trace(go.Scatter(
            x=[lx0, lx1, lx1, lx0, lx0],
            y=[ly0, ly0, ly1, ly1, ly0],
            mode="lines",
            fill="toself",
            fillcolor="rgba(0,0,0,0.001)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoveron="fills",
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── HUD overlays
    annotations.append(dict(x=10, y=height-10, text="● REC",
        font=dict(size=14, color="#ff1744", family=FONT_ORB), showarrow=False))
    annotations.append(dict(x=width-10, y=height-10, text="FPS 18.6",
        font=dict(size=13, color=TEXT_DIM, family=FONT_MON),
        showarrow=False, xanchor="right"))
    annotations.append(dict(x=10, y=10, text="twowheelersSafety  |  FUSED OVERLAY",
        font=dict(size=13, color=CYAN, family=FONT_MON),
        bgcolor="rgba(4,8,14,0.88)", showarrow=False, xanchor="left"))
    annotations.append(dict(x=width-10, y=10, text=f"FRAME #{frame_id}",
        font=dict(size=15, color=CYAN, family=FONT_MON),
        showarrow=False, xanchor="right"))

    fig.update_layout(shapes=shapes, annotations=annotations)

    return fig


_CAMERA_FIG_CACHE: Dict[str, go.Figure] = {}


def get_camera_figure(
    frame_id: str,
    objects: List[TrackedObject],
    *,
    embed_image: bool = False,
) -> go.Figure:
    """Return a cached camera figure for a frame (overlays are fixed per frame)."""
    cache_key = f"{frame_id}:{'embed' if embed_image else 'url'}"
    cached = _CAMERA_FIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fig = build_camera_figure(objects, frame_id, embed_image=embed_image)
    _CAMERA_FIG_CACHE[cache_key] = fig
    return fig


_BEV_FIG_CACHE: Dict[str, go.Figure] = {}


def get_bev_figure(frame_id: str, objects: List[TrackedObject], n_intervals: int = 0) -> go.Figure:
    """Cached 3D BEV figure keyed by frame; the radar scene is fixed."""
    cache_key = frame_id
    cached = _BEV_FIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fig = build_bev_figure(frame_id, objects, n_intervals)
    _BEV_FIG_CACHE[cache_key] = fig
    return fig


# ─── TTC / Risk Timeline ─────────────────────────────────────────────────────
def build_timeline_figure(timeline: list) -> go.Figure:
    times = [d["t"]    for d in timeline]
    ttcs  = [d["ttc"]  for d in timeline]
    risks = [d["risk"] for d in timeline]
    dists = [d["dist"] for d in timeline]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=ttcs, name="Min TTC (s)", mode="lines",
        line=dict(color="#ff1744", width=3.4),
        hovertemplate="Min TTC: %{y:.1f} s<br>Lower means less time to avoid conflict<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=risks, name="Risk Score", mode="lines",
        line=dict(color="#ffab00", width=3.0),
        hovertemplate="Risk score: %{y:.1f}<br>1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=dists, name="Closest (m)", mode="lines",
        line=dict(color=CYAN, width=2.8, dash="dash"),
        hovertemplate="Closest object distance: %{y:.1f} m<extra></extra>",
    ))

    visible_ticks = times[::5]
    fig.update_layout(**_base_layout(
        autosize=True,
        margin=dict(l=46, r=12, t=10, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#9fc0d8", family=FONT_MON),
                   tickvals=visible_ticks, ticktext=visible_ticks, color="#9fc0d8"),
        yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=0.5,
                   tickfont=dict(size=13, color="#9fc0d8"), zeroline=False),
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                    font=dict(size=13, color="#c3d8e8", family=FONT_MON),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    ))
    return fig


# ─── Semicircle Gauge ────────────────────────────────────────────────────────
def build_gauge(value: float, max_val: float, unit: str,
                level: str, label: str = "", sub: str = "",
                valueformat: str = ".1f",
                compact_number: bool = False) -> go.Figure:
    """
    Plotly Indicator gauge matching the React semicircle aesthetic.
    level: 'crit' | 'warn' | 'caut' | 'ok'.
    The KPI title/subtitle are rendered as static HTML above the gauge in the
    layout (see `_gauge_cell`), so no title is drawn inside the figure — this
    avoids Plotly's flaky two-line indicator titles and keeps them always clear.
    """
    level_colors = {"crit":"#ff1744","warn":"#ffab00","caut":"#ffd600","ok":"#00e676"}
    col = level_colors.get(level, "#00e676")

    if not math.isfinite(value):
        value = max_val
    num_size = 24 if compact_number else 30
    suffix_size = 13 if compact_number else 17
    bar_thickness = 0.24 if compact_number else 0.28
    domain_y0 = 0.14 if compact_number else 0.10
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        number=dict(
            suffix=(
                f"<span style='font-size:{suffix_size}px;color:#cfe6f7;"
                f"font-weight:700'> {unit}</span>"
            ),
            font=dict(size=num_size, color=col, family=FONT_ORB),
            valueformat=valueformat,
        ),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=0,
                      tickcolor="rgba(0,0,0,0)",
                      tickfont=dict(size=8, color="#6a8aa4")),
            bar=dict(color=col, thickness=bar_thickness),
            bgcolor="#0f1d30",
            borderwidth=0,
            steps=[
                dict(range=[0, max_val*0.33], color="rgba(21,36,54,0.6)"),
                dict(range=[max_val*0.33, max_val*0.66], color="rgba(21,36,54,0.4)"),
                dict(range=[max_val*0.66, max_val], color="rgba(21,36,54,0.2)"),
            ],
        ),
        domain=dict(x=[0.02, 0.98], y=[domain_y0, 1.0]),
    ))
    fig.update_layout(**_base_layout(
        height=112,
        margin=dict(l=4, r=4, t=6 if compact_number else 4, b=4 if compact_number else 2),
    ))
    return fig

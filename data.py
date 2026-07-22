"""
data.py — Data models, simulation engine, and safety metric calculations.
Pure Python: no Dash or Plotly dependencies.
Loads KITTI labels, radar detections, and calibration for accurate object tracking.
"""

import base64
import csv
import datetime as _dt
import math
import os
from io import BytesIO
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from PIL import Image

# ─── Dataset path (relative to this project by default) ───────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "data_sample_v1")
KITTI_BASE_PATH = os.path.abspath(
    os.environ.get("KITTI_BASE_PATH", _DEFAULT_DATA_DIR)
)
RADAR_ASSOCIATION_MARGIN_M = 0.05
IGNORE_RADAR_Z = True


def _find_data_directory(*names: str) -> str:
    """Return the first existing dataset subdirectory from the given names."""
    for name in names:
        candidate = os.path.join(KITTI_BASE_PATH, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(KITTI_BASE_PATH, names[0])


IMAGE_PATH = _find_data_directory("image", "image_2", "images", "camera")
LABEL_PATH = _find_data_directory("label_2")
CALIB_PATH = _find_data_directory("calib", "calibration")
RADAR_PATH = _find_data_directory("radar", "pc_bin", "pointcloud")
SHARED_RADAR_FILE = os.path.join(RADAR_PATH, "pc.bin")
MANIFEST_PATH = os.path.join(KITTI_BASE_PATH, "manifest.csv")

if not os.path.isdir(KITTI_BASE_PATH):
    print(
        f"[data] Dataset folder not found:\n"
        f"  {KITTI_BASE_PATH}\n"
        f"  Place KITTI-style data in ./data_sample_v1 "
        f"(image_2/, calib/, label_2/, radar/) "
        f"or set KITTI_BASE_PATH.",
        flush=True,
    )
elif not os.path.isdir(IMAGE_PATH):
    print(
        f"[data] No image folder under {KITTI_BASE_PATH}. "
        f"Expected image_2/ (or image/).",
        flush=True,
    )


def _list_files(directory: str) -> List[str]:
    """Return directory entries without failing when optional data is absent."""
    return os.listdir(directory) if os.path.isdir(directory) else []


def _radar_file_for_frame(frame_id: str) -> str:
    """Use a frame-specific radar file, or the shared radar/pc.bin file."""
    frame_file = os.path.join(RADAR_PATH, f"{frame_id}.bin")
    return frame_file if os.path.exists(frame_file) else SHARED_RADAR_FILE

# The converted dataset numbers images/labels/radar from 1, while calibration
# files are zero-based and padded (frame 1 -> 00000.txt). Prefer the manifest
# mapping so the loader also works if calibration filenames change later.
calib_source_by_frame = {}
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8-sig") as manifest_file:
        for row in csv.DictReader(manifest_file):
            sequence = (row.get("sequence") or "").strip()
            calib_source = (row.get("calib_source") or "").strip()
            if sequence and calib_source:
                calib_source_by_frame[sequence] = calib_source

# Get all available frames with matching image, label, radar, and mapped calib.
image_files_by_frame = {
    os.path.splitext(filename)[0]: filename
    for filename in _list_files(IMAGE_PATH)
    if os.path.splitext(filename)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp"}
}
image_frames = set(image_files_by_frame)
label_frames = {f.replace('.txt', '') for f in _list_files(LABEL_PATH) if f.endswith('.txt')}
radar_frames = {
    f.replace('.bin', '')
    for f in _list_files(RADAR_PATH)
    if f.endswith('.bin') and f.lower() != "pc.bin"
}
calib_files = {f for f in _list_files(CALIB_PATH) if f.endswith('.txt')}

frames_with_radar = image_frames & label_frames
if not os.path.exists(SHARED_RADAR_FILE):
    frames_with_radar &= radar_frames

calib_file_by_frame = {}
for frame_id in frames_with_radar:
    mapped_name = calib_source_by_frame.get(frame_id)
    same_name = f"{frame_id}.txt"
    zero_based_name = f"{int(frame_id) - 1:05d}.txt" if frame_id.isdigit() else ""
    for candidate in (mapped_name, same_name, zero_based_name):
        if candidate and candidate in calib_files:
            calib_file_by_frame[frame_id] = candidate
            break

frame_candidates = sorted(
    calib_file_by_frame,
    key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
)

RISK_COLORS = {
    "CRITICAL": "#ff1744",
    "HIGH":     "#ffab00",
    "MEDIUM":   "#ffd600",
    "LOW":      "#00e676",
}
# Class colors mirrored from radar_annotator_v3 (views/annotation_classes.py)
# so the dashboard matches the annotation tool's palette exactly.
OBJECT_CLASS_COLORS = {
    "Car": "#1d4ed8",
    "Pedestrian": "#15803d",
    "Cyclist": "#0f766e",
    "Rider": "#1e40af",
    "Bicycle": "#0e7490",
    "Motorcycle": "#b91c1c",
    "Motorbike": "#b91c1c",
    "E-Scooter / Moped": "#991b1b",
    "Truck": "#b45309",
    "Van": "#374151",
    "Bus": "#a16207",
    "Animal": "#713f12",
    "Other vehicle": "#475569",
    "Other": "#64748b",
}

# Per-object risk thresholds — TTC and distance are evaluated independently;
# the higher (more severe) tier wins. Path conflict bumps one level.
RISK_TTC_THRESHOLDS = {
    "CRITICAL": 1.0,
    "HIGH":     2.0,
    "MEDIUM":   3.5,
}
RISK_DIST_THRESHOLDS = {
    "CRITICAL": 2.0,
    "HIGH":     5.0,
    "MEDIUM":   10.0,
}
CROWD_AHEAD_FORWARD_M = 45.0
CROWD_AHEAD_LATERAL_M = 12.0
CROWD_AHEAD_WARN_COUNT = 4
RISK_LEVEL_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
RISK_LEVEL_RANK = {level: idx for idx, level in enumerate(RISK_LEVEL_ORDER)}

EGO_SPEED_MS = 32.0 / 3.6  # 8.89 m/s — assumed ego speed (matches status-bar display)

# ─── KITTI Data Parsing Functions ────────────────────────────────────────────
def load_kitti_labels_raw(frame_id: str) -> List[str]:
    """Load raw KITTI label lines for detailed parsing."""
    label_file = os.path.join(LABEL_PATH, f"{frame_id}.txt")
    lines = []

    if not os.path.exists(label_file):
        return lines

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    return lines


def _split_kitti_label_line(line: str) -> Optional[dict]:
    """Parse one KITTI label line; class name may span multiple tokens."""
    tok = line.strip().split()
    if len(tok) < 15:
        return None

    numeric = tok[-14:]
    try:
        return {
            'type': ' '.join(tok[:-14]),
            'truncated': float(numeric[0]),
            'occluded': int(float(numeric[1])),
            'alpha': float(numeric[2]),
            'bbox': [float(numeric[i]) for i in range(3, 7)],
            'dimensions': [float(numeric[i]) for i in range(7, 10)],
            'location': [float(numeric[i]) for i in range(10, 13)],
            'rotation_y': float(numeric[13]),
        }
    except (ValueError, IndexError):
        return None


def load_kitti_labels(frame_id: str) -> List[dict]:
    """Load KITTI detection labels for a frame."""
    label_file = os.path.join(LABEL_PATH, f"{frame_id}.txt")
    detections = []

    if not os.path.exists(label_file):
        return detections

    with open(label_file, 'r') as f:
        for line in f:
            detection = _split_kitti_label_line(line)
            if detection is not None:
                detections.append(detection)

    return detections


def load_kitti_calib(frame_id: str) -> Dict[str, np.ndarray]:
    """Load and validate KITTI calibration matrices for a frame."""
    calib_name = (
        calib_file_by_frame.get(frame_id)
        or calib_source_by_frame.get(frame_id)
        or f"{frame_id}.txt"
    )
    calib_file = os.path.join(CALIB_PATH, calib_name)
    calib = {}

    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"Calibration file not found for frame {frame_id}: {calib_file}"
        )

    try:
        with open(calib_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#'):
                    fields = stripped[1:].split()
                    if len(fields) == 3 and fields[0].lower() == 'image_size':
                        calib['image_size'] = np.array(
                            [int(fields[1]), int(fields[2])], dtype=np.int64
                        )
                    continue
                if ':' in line:
                    key, values = line.split(':', 1)
                    values = values.strip()
                    if values:
                        calib[key.strip()] = np.array(
                            [float(x) for x in values.split()], dtype=np.float64
                        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not parse calibration {calib_file}: {exc}"
        ) from exc

    validate_kitti_calib(calib, calib_file)
    return calib


def validate_kitti_calib(
    calib: Dict[str, np.ndarray],
    source: str = "calibration",
) -> None:
    """Reject incomplete or physically invalid KITTI calibration values."""
    projection = next(
        (
            calib[key][:12].reshape(3, 4)
            for key in ('P2', 'P1', 'P0', 'P3')
            if key in calib and calib[key].size >= 12
        ),
        None,
    )
    transform = next(
        (
            calib[key][:12].reshape(3, 4)
            for key in ('Tr_velo_to_cam', 'Tr_radar_to_cam', 'Tr_lidar_to_cam')
            if key in calib and calib[key].size >= 12
        ),
        None,
    )
    errors = []
    if projection is None:
        errors.append("missing P0/P1/P2/P3 projection matrix")
    elif not np.isfinite(projection).all():
        errors.append("projection matrix contains non-finite values")
    elif projection[0, 0] <= 0 or projection[1, 1] <= 0:
        errors.append("projection focal lengths must be positive")

    if transform is None:
        errors.append("missing radar/velo-to-camera transform")
    elif not np.isfinite(transform).all():
        errors.append("extrinsic transform contains non-finite values")

    image_size = calib.get('image_size')
    if image_size is None or image_size.size < 2:
        errors.append("missing '# image_size WIDTH HEIGHT' metadata")
    elif (
        not np.isfinite(image_size[:2]).all()
        or np.any(image_size[:2] <= 0)
    ):
        errors.append("image_size must be positive and finite")

    if transform is not None and np.isfinite(transform).all():
        rotation = transform[:, :3]
        if 'R0_rect' in calib:
            if calib['R0_rect'].size < 9:
                errors.append("R0_rect must contain 9 values")
            else:
                rectification = calib['R0_rect'][:9].reshape(3, 3)
                if not np.isfinite(rectification).all():
                    errors.append("R0_rect contains non-finite values")
                else:
                    rotation = rectification @ rotation
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            errors.append("extrinsic rotation is not orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-3):
            errors.append(
                f"extrinsic rotation determinant must be +1 (got {determinant:.6f})"
            )

    if errors:
        raise ValueError(f"Invalid calibration {source}: {'; '.join(errors)}")


def load_radar_points(frame_id: str) -> np.ndarray:
    """Load raw radar .bin file (N×7: x, y, z, RCS, v_r, v_r_comp, time)."""
    radar_file = _radar_file_for_frame(frame_id)
    if not os.path.exists(radar_file):
        return np.array([]).reshape(0, 7)

    pts = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
    # Drop rows with non-finite XYZ
    finite = np.all(np.isfinite(pts[:, :3]), axis=1)
    return pts[finite]


def get_T_cam_from_master(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """4×4 transform: master (ego/velo) frame → camera frame."""
    T = np.eye(4, dtype=np.float64)
    transform = None
    for key in ('Tr_velo_to_cam', 'Tr_radar_to_cam', 'Tr_lidar_to_cam'):
        if key in calib and calib[key].size >= 12:
            transform = calib[key][:12].reshape(3, 4)
            break
    if transform is None:
        raise ValueError("Calibration has no radar/velo-to-camera transform")

    T[:3, :] = transform
    if 'R0_rect' in calib and calib['R0_rect'].size >= 9:
        rectified = np.eye(4, dtype=np.float64)
        rectified[:3, :3] = calib['R0_rect'][:9].reshape(3, 3)
        T = rectified @ T
    return T


def get_T_master_from_cam(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """4×4 transform: camera frame → master (ego) frame (inverse of above)."""
    return np.linalg.inv(get_T_cam_from_master(calib))


def get_K_from_calib(
    calib: Dict[str, np.ndarray],
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Extract 3×3 intrinsic matrix from P2 in KITTI calibration."""
    projection = None
    for key in ('P2', 'P1', 'P0', 'P3'):
        if key in calib and calib[key].size >= 12:
            projection = calib[key][:12].reshape(3, 4)
            break
    if projection is None:
        raise ValueError("Calibration has no P0/P1/P2/P3 projection matrix")

    K = projection[:3, :3].copy()
    calib_size = calib.get('image_size')
    if image_size is not None and calib_size is not None and calib_size.size >= 2:
        source_width, source_height = map(float, calib_size[:2])
        target_width, target_height = map(float, image_size)
        if source_width > 0 and source_height > 0:
            K[0, :] *= target_width / source_width
            K[1, :] *= target_height / source_height
    return K


def box_8_corners(center: np.ndarray, L: float, W: float,
                   H: float, yaw: float) -> np.ndarray:
    """Return 8 corners of a 3D box in master frame, shape (8, 3).

    Corner ordering (bottom=z−H/2, top=z+H/2):
      0–3: bottom face  rear-right, front-right, front-left, rear-left
      4–7: same order, top face
    """
    c, s = math.cos(yaw), math.sin(yaw)
    # Local-frame offsets
    lxs = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=np.float64) * L * 0.5
    lys = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=np.float64) * W * 0.5
    lzs = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=np.float64) * H * 0.5
    rot_x = c * lxs - s * lys + center[0]
    rot_y = s * lxs + c * lys + center[1]
    rot_z = lzs + center[2]
    return np.stack([rot_x, rot_y, rot_z], axis=1)


def project_to_image(corners_master: np.ndarray,
                      T_cam_from_master: np.ndarray,
                      K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 8 master-frame corners → image UV.

    Returns (uvs shape (8,2), depths shape (8,)).
    Points behind the camera have depth ≤ 0 and uv = (−1, −1).
    """
    uvs   = np.full((8, 2), -1.0, dtype=np.float64)
    depths = np.zeros(8, dtype=np.float64)
    for i, corner in enumerate(corners_master):
        cam = (T_cam_from_master @ np.append(corner, 1.0))[:3]
        depths[i] = cam[2]
        if cam[2] > 0:
            px = K @ cam
            uvs[i] = [px[0] / px[2], px[1] / px[2]]
    return uvs, depths


def parse_kitti_line(line: str, T_master_from_cam: np.ndarray) -> Optional[Dict]:
    """Parse one KITTI label line into box attributes in master frame."""
    parsed = _split_kitti_label_line(line)
    if parsed is None:
        return None

    try:
        cls = parsed['type']
        # KITTI format: 8=H, 9=W, 10=L, 11=cx_cam, 12=cy_cam, 13=cz_cam, 14=rotation_y
        H, W, L = parsed['dimensions']
        cx, cy, cz = parsed['location']
        rotation = parsed['rotation_y']

        # Camera-frame center (bottom face)
        bottom_cam = np.array([cx, cy, cz, 1.0])
        bottom_mast = (T_master_from_cam @ bottom_cam)[:3]

        # Box centre is half-height above bottom face in master frame (z up)
        center = bottom_mast + np.array([0.0, 0.0, H * 0.5])

        # Yaw in master frame: undo the wrap_pi(-yaw) applied at export
        yaw = -rotation

        return {
            'cls': cls,
            'center': center,
            'L': L,           # length (forward direction)
            'W': W,           # width (left direction)
            'H': H,           # height (up direction)
            'yaw': yaw,       # heading in master frame (radians)
            'bbox': parsed['bbox'],  # 2D bbox in image
            'occluded': parsed['occluded'],
            'rotation': rotation  # original KITTI rotation_y (for reference)
        }
    except (ValueError, IndexError):
        return None


def points_inside_box(
    pts_xyz: np.ndarray,
    box: Dict,
    *,
    ignore_z: bool = IGNORE_RADAR_Z,
    margin_xy: float = RADAR_ASSOCIATION_MARGIN_M,
) -> np.ndarray:
    """
    Boolean mask of radar points inside a 3D box.
    Uses box-local axis-aligned range checks after rotation.
    """
    center = box['center']
    L, W, H = box['L'], box['W'], box['H']
    yaw = box['yaw']

    # Translate to box-local origin
    rel = pts_xyz - center  # (N, 3)

    # Rotate around Z by -yaw to align with box axes
    c = math.cos(yaw)
    s = math.sin(yaw)
    lx = c * rel[:, 0] + s * rel[:, 1]   # along length
    ly = -s * rel[:, 0] + c * rel[:, 1]  # along width
    lz = rel[:, 2]                        # up

    margin = max(float(margin_xy), 0.0)
    inside = (
        (np.abs(lx) <= L * 0.5 + margin)
        & (np.abs(ly) <= W * 0.5 + margin)
    )
    if not ignore_z:
        inside &= np.abs(lz) <= H * 0.5
    return inside


def extract_velocity_from_radar(inside_pts: np.ndarray) -> Dict:
    """
    Extract velocity attributes from radar points inside a box.
    Columns: 4=v_r (raw), 5=v_r_comp (ego-compensated).
    Uses median (robust to outliers).
    """
    if inside_pts.shape[0] == 0:
        return {
            "n_pts": 0,
            "v_r": None,
            "v_r_comp": None,
            "v_r_std": None,
            "v_r_comp_std": None,
        }

    vr = inside_pts[:, 4]
    vr_comp = inside_pts[:, 5]

    def robust_median_std(arr):
        f = arr[np.isfinite(arr)]
        if f.size == 0:
            return None, None
        med = float(np.median(f))
        std = float(np.std(f)) if f.size > 1 else 0.0
        return med, std

    vr_med, vr_std = robust_median_std(vr)
    vrcomp_med, vrcomp_std = robust_median_std(vr_comp)

    return {
        "n_pts": int(inside_pts.shape[0]),
        "v_r": vr_med,
        "v_r_std": vr_std,
        "v_r_comp": vrcomp_med,
        "v_r_comp_std": vrcomp_std,
    }



def yaw_to_direction(yaw_rad: float) -> str:
    """Convert object yaw (master frame, rad) to a short heading description.

    Master frame: x = forward, y = left.  yaw=0 → heading same as ego.
    """
    deg = ((math.degrees(yaw_rad) + 180) % 360) - 180  # −180…180
    if abs(deg) <= 22:
        return "↑ With traffic"
    if abs(deg) >= 158:
        return "↓ Oncoming"
    if 22 < deg <= 112:
        return "← Crossing L"
    if -112 <= deg < -22:
        return "→ Crossing R"
    if deg > 112:
        return "↖ Turning back"
    return "↗ Turning back"


_FRAME_CACHE: Dict[str, dict] = {}
_WEATHER_CACHE: Dict[str, dict] = {}
_OBJECTS_BY_INDEX: Dict[int, List["TrackedObject"]] = {}


def _encode_image_data_uri(img: Image.Image) -> str:
    """Encode a PIL image once for Plotly layout_image (avoids re-serializing each tick)."""
    buf = BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_kitti_frame(frame_id: str) -> dict:
    """Load all KITTI data for a frame: labels, radar, calibration, image."""
    cached = _FRAME_CACHE.get(frame_id)
    if cached is not None:
        return cached

    image_filename = image_files_by_frame.get(frame_id, f"{frame_id}.png")
    calib = load_kitti_calib(frame_id)
    frame_data = {
        'frame_id': frame_id,
        'image_path': os.path.join(IMAGE_PATH, image_filename),
        'labels':    load_kitti_labels(frame_id),
        'raw_lines': load_kitti_labels_raw(frame_id),
        'radar_pts': load_radar_points(frame_id),
        'calib':     calib,
        'radar_path': _radar_file_for_frame(frame_id),
    }
    if os.path.exists(frame_data['image_path']):
        with Image.open(frame_data['image_path']) as img:
            frame_data['image_size'] = img.size
    elif 'image_size' in calib and calib['image_size'].size >= 2:
        frame_data['image_size'] = tuple(map(int, calib['image_size'][:2]))
    _FRAME_CACHE[frame_id] = frame_data
    return frame_data


def get_frame_image_path(frame_id: str) -> str:
    """Return the camera image path for a playable frame."""
    image_filename = image_files_by_frame.get(frame_id, f"{frame_id}.png")
    return os.path.join(IMAGE_PATH, image_filename)


_IMAGE_DATA_URI_CACHE: Dict[str, str] = {}


def get_frame_image_data_uri(frame_id: str) -> Optional[str]:
    """Inline JPEG data-URI for static HTML/video export (no Flask server needed)."""
    cached = _IMAGE_DATA_URI_CACHE.get(frame_id)
    if cached is not None:
        return cached
    image_path = get_frame_image_path(frame_id)
    if not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            uri = _encode_image_data_uri(img)
    except OSError:
        return None
    _IMAGE_DATA_URI_CACHE[frame_id] = uri
    return uri


def infer_weather_from_image(frame_id: Optional[str] = None) -> Dict[str, object]:
    """Infer coarse weather/visibility risk from the camera frame itself.

    The thresholds are intentionally strict so normal daylight, shade, and camera
    exposure changes do not become false weather alerts.
    """
    if frame_id is None:
        if not AVAILABLE_FRAMES:
            return {
                "label": "Unknown",
                "level": "ok",
                "score": 0,
                "detail": "No camera frames available",
            }
        frame_id = AVAILABLE_FRAMES[current_frame_index % TOTAL_FRAMES]

    cached = _WEATHER_CACHE.get(frame_id)
    if cached is not None:
        return cached

    image_path = get_frame_image_path(frame_id)
    if not os.path.exists(image_path):
        result = {
            "label": "Unknown",
            "level": "ok",
            "score": 0,
            "detail": "Camera image missing",
        }
        _WEATHER_CACHE[frame_id] = result
        return result

    try:
        with Image.open(image_path) as img:
            resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            img = img.convert("RGB")
            img.thumbnail((192, 144), resample)
            arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:
        result = {
            "label": "Unknown",
            "level": "ok",
            "score": 0,
            "detail": f"Weather inference failed: {exc}",
        }
        _WEATHER_CACHE[frame_id] = result
        return result

    if arr.size == 0:
        result = {
            "label": "Unknown",
            "level": "ok",
            "score": 0,
            "detail": "Camera image is empty",
        }
        _WEATHER_CACHE[frame_id] = result
        return result

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = np.where(maxc > 1e-6, (maxc - minc) / maxc, 0.0)

    brightness = float(np.mean(luma))
    contrast = float(np.std(luma))
    sat_mean = float(np.mean(saturation))
    dark_frac = float(np.mean(luma < 45.0))
    bright_frac = float(np.mean(luma > 238.0))

    score = 0

    # Independent visibility-condition evidence. Conditions can co-occur (e.g.
    # night + rain), so each is a separate boolean rather than one exclusive tag.
    is_night = brightness < 48.0 or (brightness < 65.0 and contrast < 30.0)
    is_rain = (brightness < 105.0 and contrast < 32.0
               and sat_mean < 0.24 and dark_frac > 0.12)
    is_fog = contrast < 24.0 and sat_mean < 0.22 and brightness < 140.0
    is_glare = bright_frac > 0.34 and contrast > 55.0

    if is_night:
        score = max(score, 88 if brightness < 48.0 else 72)
    if is_fog:
        score = max(score, 82 if (contrast < 18.0 and sat_mean < 0.18) else 62)
    if is_rain:
        score = max(score, 66)
    if is_glare:
        score = max(score, 58)

    time_of_day = "night" if is_night else "day"

    # Compose a concise, human label. Rain/fog are the actionable hazards, so
    # they lead; night is prepended so combined scenes read as "Night Rain" etc.
    if is_rain:
        hazard = "Rain"
    elif is_fog:
        hazard = "Fog"
    elif is_glare:
        hazard = "Glare"
    else:
        hazard = ""

    if is_night and hazard:
        label = f"Night {hazard}"
    elif is_night:
        label = "Night"
    elif hazard:
        label = hazard
    else:
        label = "Clear"

    level = "crit" if score >= 80 else "warn" if score >= 55 else "caut" if score else "ok"
    result = {
        "label": label,
        "level": level,
        "score": int(score),
        "time_of_day": time_of_day,
        "is_night": is_night,
        "is_rain": is_rain,
        "is_fog": is_fog,
        "is_glare": is_glare,
        "detail": (
            f"Image weather inference: brightness {brightness:.0f}, "
            f"contrast {contrast:.0f}, saturation {sat_mean:.2f}, "
            f"dark {dark_frac * 100:.0f}%, glare {bright_frac * 100:.0f}%"
        ),
    }
    _WEATHER_CACHE[frame_id] = result
    return result


# Scene clock: the displayed time-of-day should match the lighting of the
# footage rather than the wall clock, so a night sequence reads as evening/night.
_SCENE_CLOCK_ANCHOR: Optional[_dt.datetime] = None
_SCENE_CLOCK_REALSTART: Optional[_dt.datetime] = None


def _init_scene_clock(now: _dt.datetime) -> None:
    global _SCENE_CLOCK_ANCHOR, _SCENE_CLOCK_REALSTART
    first_frame = AVAILABLE_FRAMES[0] if AVAILABLE_FRAMES else None
    weather = infer_weather_from_image(first_frame)
    time_of_day = str(weather.get("time_of_day", "day"))
    if time_of_day == "night":
        anchor = now.replace(hour=20, minute=42, second=0, microsecond=0)
    else:
        anchor = now.replace(hour=13, minute=12, second=0, microsecond=0)
    _SCENE_CLOCK_ANCHOR = anchor
    _SCENE_CLOCK_REALSTART = now


def get_scene_clock(now: Optional[_dt.datetime] = None) -> _dt.datetime:
    """Wall-clock-ticking time anchored to the scene's day/night lighting.

    The hour reflects whether the first loaded frame is day or night; the clock
    then advances in real time so the displayed seconds keep moving.
    """
    if now is None:
        now = _dt.datetime.now()
    if _SCENE_CLOCK_ANCHOR is None or _SCENE_CLOCK_REALSTART is None:
        _init_scene_clock(now)
    return _SCENE_CLOCK_ANCHOR + (now - _SCENE_CLOCK_REALSTART)


def _frame_is_playable(frame_id: str) -> bool:
    """Frame must have labels and a validated per-frame calibration file."""
    if len(load_kitti_labels(frame_id)) == 0:
        return False
    if frame_id not in calib_file_by_frame:
        return False
    try:
        load_kitti_calib(frame_id)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


# Only keep frames with labels and validated per-frame calibration
AVAILABLE_FRAMES = [frame_id for frame_id in frame_candidates if _frame_is_playable(frame_id)]
TOTAL_FRAMES = len(AVAILABLE_FRAMES)


# Per-frame calibration is already validated while building AVAILABLE_FRAMES.
# Images and radar load on first access via _FRAME_CACHE (no eager warm — 400+ frames
# would JPEG-encode every image at import and block startup for tens of seconds).


def get_frame_calib_name(frame_id: str) -> str:
    return calib_file_by_frame.get(frame_id, f"{frame_id}.txt")


def get_frame_info(frame_index: int) -> Dict[str, str]:
    if not AVAILABLE_FRAMES:
        return {"frame_id": "—", "index": "0", "total": "0", "calib": "—"}
    idx = frame_index % TOTAL_FRAMES
    frame_id = AVAILABLE_FRAMES[idx]
    return {
        "frame_id": frame_id,
        "index": str(idx + 1),
        "total": str(TOTAL_FRAMES),
        "calib": get_frame_calib_name(frame_id),
    }

# ─── Tracked-object dataclass ─────────────────────────────────────────────────
@dataclass
class TrackedObject:
    id:           str
    cls:          str
    source:       str    # Fused | Radar | Camera
    dist:         float  # metres from ego (ground plane)
    rel_vel:      float  # m/s  (negative = approaching)
    ttc:          float  # seconds
    req_decel:    float  # m/s²
    occupancy:    float  # 0–1
    path_conflict: bool
    confidence:   float  # 0–1
    risk:         str    # CRITICAL | HIGH | MEDIUM | LOW
    bev_x:        float  # BEV lateral (m, master y)
    bev_y:        float  # BEV forward (m, master x)
    cam_cx:       float  # 2D bbox centre x (pixel); -1 = not visible
    cam_cy:       float  # 2D bbox centre y (pixel)
    cam_w:        float  # 2D bbox width (pixel)
    cam_h:        float  # 2D bbox height (pixel)
    # 3D box attributes for oriented BEV box and camera wireframe
    box_l:   float = 4.0   # length along heading (m)
    box_w:   float = 1.8   # width  (m)
    box_h:   float = 1.5   # height (m)
    box_z:   float = 0.75  # z-centre in master frame (m)
    box_yaw: float = 0.0   # heading in master frame (rad)
    heading: str   = "—"   # human-readable heading description
    risk_reason:  str = ""  # human-readable trigger(s) for assigned risk
    occluded:     int = 0   # KITTI/VoD occlusion: 2 = not in camera (radar-only)


def is_radar_only_object(occluded: int) -> bool:
    """True when the annotation marks the object as not visible in the camera."""
    return int(occluded) == 2


# ─── Risk classification ─────────────────────────────────────────────────────
def _risk_from_ttc(ttc: float) -> str:
    if math.isfinite(ttc):
        if ttc < RISK_TTC_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        if ttc < RISK_TTC_THRESHOLDS["HIGH"]:
            return "HIGH"
        if ttc < RISK_TTC_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
    return "LOW"


def _risk_from_distance(dist: float) -> str:
    if dist < RISK_DIST_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if dist < RISK_DIST_THRESHOLDS["HIGH"]:
        return "HIGH"
    if dist < RISK_DIST_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def _max_risk_level(a: str, b: str) -> str:
    return a if RISK_LEVEL_RANK[a] >= RISK_LEVEL_RANK[b] else b


def _bump_risk_level(risk: str) -> str:
    idx = RISK_LEVEL_RANK[risk]
    if idx < len(RISK_LEVEL_ORDER) - 1:
        return RISK_LEVEL_ORDER[idx + 1]
    return risk


def classify_object_risk(
    ttc: float,
    dist: float,
    *,
    path_conflict: bool = False,
    rel_vel: float = 0.0,
) -> Tuple[str, str]:
    """Combine TTC + distance tiers; only bump for path conflicts when the object is truly relevant."""
    ttc_risk = _risk_from_ttc(ttc)
    dist_risk = _risk_from_distance(dist)
    risk = _max_risk_level(ttc_risk, dist_risk)

    reasons = []
    if math.isfinite(ttc) and ttc < RISK_TTC_THRESHOLDS["MEDIUM"]:
        reasons.append(f"TTC {ttc:.1f}s (<{RISK_TTC_THRESHOLDS['MEDIUM']}s)")
    if dist < RISK_DIST_THRESHOLDS["MEDIUM"]:
        reasons.append(f"dist {dist:.1f}m (<{RISK_DIST_THRESHOLDS['MEDIUM']}m)")

    if (
        not path_conflict
        and dist >= RISK_DIST_THRESHOLDS["MEDIUM"]
        and (not math.isfinite(ttc) or ttc >= RISK_TTC_THRESHOLDS["MEDIUM"])
        and rel_vel >= -0.3
    ):
        return "LOW", "all metrics clear"

    if path_conflict and 0 < dist and (
        dist < RISK_DIST_THRESHOLDS["MEDIUM"]
        or (math.isfinite(ttc) and ttc < RISK_TTC_THRESHOLDS["MEDIUM"])
        or rel_vel < -0.3
    ):
        risk = _bump_risk_level(risk)
        reasons.append("ego-path conflict")

    if not reasons:
        reasons.append("all metrics clear")

    return risk, " · ".join(reasons)


def risk_threshold_legend() -> List[Tuple[str, str]]:
    return [
        ("CRITICAL", f"TTC < {RISK_TTC_THRESHOLDS['CRITICAL']}s  OR  dist < {RISK_DIST_THRESHOLDS['CRITICAL']}m"),
        ("HIGH",     f"TTC < {RISK_TTC_THRESHOLDS['HIGH']}s  OR  dist < {RISK_DIST_THRESHOLDS['HIGH']}m"),
        ("MEDIUM",   f"TTC < {RISK_TTC_THRESHOLDS['MEDIUM']}s  OR  dist < {RISK_DIST_THRESHOLDS['MEDIUM']}m"),
        ("LOW",      f"TTC >= {RISK_TTC_THRESHOLDS['MEDIUM']}s and dist >= {RISK_DIST_THRESHOLDS['MEDIUM']}m"),
    ]


# ─── Initial scene ────────────────────────────────────────────────────────────
def _ego_motion_from_radar_points(radar_pts: np.ndarray) -> Optional[float]:
    """Estimate per-frame ego speed from raw vs ego-compensated Doppler."""
    if radar_pts is None or radar_pts.size == 0 or radar_pts.shape[1] <= 5:
        return None
    raw = radar_pts[:, 4].astype(np.float64)
    comp = radar_pts[:, 5].astype(np.float64)
    delta = np.abs(comp - raw)
    valid = delta[np.isfinite(delta) & (delta > 0.05) & (delta < 45.0)]
    if valid.size < 20:
        return None
    return float(np.median(valid))


def get_frame_ego_motion_ms(frame_id: Optional[str] = None,
                            radar_pts: Optional[np.ndarray] = None) -> float:
    if radar_pts is None and frame_id is not None:
        radar_pts = load_radar_points(frame_id)
    estimated = _ego_motion_from_radar_points(radar_pts) if radar_pts is not None else None
    return estimated if estimated is not None else EGO_SPEED_MS


def get_ego_motion_avg_kmh(frame_id: Optional[str] = None,
                           radar_pts: Optional[np.ndarray] = None) -> float:
    return get_frame_ego_motion_ms(frame_id, radar_pts) * 3.6


def crowding_ahead(objects: List["TrackedObject"]) -> Dict[str, float]:
    ahead = [
        o for o in objects
        if 0.0 < o.bev_y <= CROWD_AHEAD_FORWARD_M
        and abs(o.bev_x) <= CROWD_AHEAD_LATERAL_M
    ]
    nearest = float(min((o.bev_y for o in ahead), default=float("inf")))
    return {
        "count": len(ahead),
        "nearest": nearest,
        "warning": len(ahead) >= CROWD_AHEAD_WARN_COUNT,
        "score": min(100, round(len(ahead) / CROWD_AHEAD_WARN_COUNT * 100)),
    }


def _build_tracked_object(i: int, box: dict, radar_pts: np.ndarray,
                          ego_speed_ms: float) -> TrackedObject:
    """Shared object-construction logic used by both init and update functions."""
    obj_id    = f"{i+1:02d}"
    center    = box['center']
    L, W, H   = box['L'], box['W'], box['H']
    yaw       = box['yaw']

    dist = math.sqrt(center[0]**2 + center[1]**2)

    # Radar velocity: points inside this box
    if radar_pts.shape[0] > 0:
        mask   = points_inside_box(radar_pts[:, :3], box)
        inside = radar_pts[mask]
    else:
        inside = radar_pts  # empty (0, 7)
    vel_info = extract_velocity_from_radar(inside)
    # Always read the object's radial velocity from the ego-motion-compensated
    # channel (v_r_comp). This is the value used for display, closing-rate TTC,
    # required deceleration and risk classification.
    rel_vel = vel_info['v_r_comp'] if vel_info['v_r_comp'] is not None else 0.0

    # TTC — two contributions, use the more conservative (shorter) one:
    #   1. Radar closing rate from the compensated velocity: rel_vel < 0 means
    #      the target is actively closing.
    #   2. Ego-approach: even a stationary forward object will be reached at ego speed.
    if rel_vel < -0.3:
        ttc_radar = dist / abs(rel_vel)
    else:
        ttc_radar = float('inf')

    forward_dist = max(0.0, float(center[0]))   # positive = ahead in master frame
    ttc_ego = (forward_dist / max(ego_speed_ms, 0.1)) if forward_dist > 0.5 else float('inf')

    ttc = min(ttc_radar, ttc_ego)

    # Required deceleration to avoid the conflict. Derive the closing speed from
    # the same model as TTC (the shorter of the object's own approach and the
    # ego closing on an in-path object) so a near-stationary pedestrian the ego
    # is driving toward still produces a real brake demand — using only the
    # compensated object velocity here would read ~0 and never trigger.
    if math.isfinite(ttc) and ttc > 0.05:
        closing_speed = dist / ttc
        req_decel = (closing_speed ** 2) / (2 * max(dist - 5, 1)) if closing_speed > 0.3 else 0.0
    else:
        req_decel = 0.0

    x1, y1, x2, y2 = box['bbox']
    occluded = int(box.get('occluded', 0))
    radar_only = is_radar_only_object(occluded)
    if radar_only:
        # Match annotator behaviour: occlusion 2 = radar-only, hidden on camera.
        cam_cx = cam_cy = cam_w = cam_h = -1.0
    else:
        cam_cx = (x1 + x2) / 2
        cam_cy = (y1 + y2) / 2

    occupancy = min(1.0,
        0.22
        + max(0.0, 1.0 - abs(center[1]) / 4.0) * 0.36
        + max(0.0, 1.0 - abs(center[0]) / 28.0) * 0.28
    )
    path_conflict = abs(center[1]) < 3.5 and 0 < center[0] < 20.0
    confidence    = min(1.0, 0.5 + vel_info['n_pts'] * 0.05)
    risk, risk_reason = classify_object_risk(
        ttc,
        dist,
        path_conflict=path_conflict,
        rel_vel=rel_vel,
    )

    return TrackedObject(
        id=obj_id, cls=box['cls'], source="Radar" if radar_only else "Fused",
        dist=round(dist, 1),
        rel_vel=round(rel_vel, 2),
        ttc=round(ttc, 1) if math.isfinite(ttc) else float('inf'),
        req_decel=round(req_decel, 1),
        occupancy=round(occupancy, 2),
        path_conflict=path_conflict,
        confidence=round(confidence, 2),
        risk=risk,
        risk_reason=risk_reason,
        bev_x=round(center[1], 1),
        bev_y=round(center[0], 1),
        cam_cx=round(cam_cx) if cam_cx >= 0 else -1,
        cam_cy=round(cam_cy) if cam_cy >= 0 else -1,
        cam_w=round(x2 - x1) if (not radar_only and (x2 - x1) >= 0) else -1,
        cam_h=round(y2 - y1) if (not radar_only and (y2 - y1) >= 0) else -1,
        box_l=round(L, 2),
        box_w=round(W, 2),
        box_h=round(H, 2),
        box_z=round(float(center[2]), 2),
        box_yaw=round(yaw, 4),
        heading=yaw_to_direction(yaw),
        occluded=occluded,
    )


def load_objects_for_frame_index(frame_index: int) -> List[TrackedObject]:
    """Load tracked objects for a specific sequential frame index."""
    if not AVAILABLE_FRAMES:
        return []
    idx = frame_index % TOTAL_FRAMES
    cached = _OBJECTS_BY_INDEX.get(idx)
    if cached is not None:
        return cached

    frame_id = AVAILABLE_FRAMES[idx]
    frame_data = load_kitti_frame(frame_id)
    T_mfc = get_T_master_from_cam(frame_data['calib'])
    radar_pts = frame_data['radar_pts']
    ego_speed_ms = get_frame_ego_motion_ms(frame_id, radar_pts)
    objects = []
    for i, line in enumerate(frame_data['raw_lines']):
        box = parse_kitti_line(line, T_mfc)
        if box is not None:
            objects.append(_build_tracked_object(i, box, radar_pts, ego_speed_ms))
    _OBJECTS_BY_INDEX[idx] = objects
    return objects


def objects_for_playback_tick(frame_index: int) -> List[TrackedObject]:
    """Load objects for an explicit sequential frame index."""
    global current_frame_index
    if not AVAILABLE_FRAMES:
        return []
    current_frame_index = frame_index % TOTAL_FRAMES
    return load_objects_for_frame_index(current_frame_index)


def get_initial_objects() -> List[TrackedObject]:
    """Load initial objects from the first available KITTI frame."""
    return load_objects_for_frame_index(0)


INITIAL_OBJECTS = get_initial_objects()

# ─── Frame tracking for KITTI sequence ────────────────────────────────────────
current_frame_index = 0
playback_tick = 0

# ─── Serialisation helpers ────────────────────────────────────────────────────
def objects_to_dicts(objects: List[TrackedObject]) -> list:
    result = []
    for o in objects:
        d = asdict(o)
        # JSON cannot represent float('inf') — replace with 9999 sentinel
        if not math.isfinite(d.get('ttc', 0)):
            d['ttc'] = 9999.0
        result.append(d)
    return result

def dicts_to_objects(dicts: list) -> List[TrackedObject]:
    result = []
    for d in dicts:
        if d.get('ttc', 0) >= 9999.0:
            d = {**d, 'ttc': float('inf')}
        d.setdefault('risk_reason', '')
        d.setdefault('occluded', 0)
        result.append(TrackedObject(**d))
    return result


# ─── Timeline bootstrap ───────────────────────────────────────────────────────
def generate_timeline(objects: List[TrackedObject]) -> list:
    closest = get_closest(objects)
    min_ttc = get_min_ttc(objects)
    risk_score = _risk_score_from_category(closest.risk)
    data = []
    for i in range(31):
        x = i - 30
        data.append({
            "t":    "now" if x == 0 else f"{x}s",
            "ttc":  min_ttc,
            "risk": risk_score,
            "dist": closest.dist,
        })
    return data


# ─── Simulation step ─────────────────────────────────────────────────────────


def update_timeline(timeline: list, objects: List[TrackedObject]) -> list:
    closest = get_closest(objects)
    min_ttc = get_min_ttc(objects)
    risk_score = _risk_score_from_category(closest.risk)

    new_pt = {
        "t":    "now",
        "ttc":  min_ttc,
        "risk": risk_score,
        "dist": closest.dist,
    }
    new_tl = timeline[1:] + [new_pt]
    for i, pt in enumerate(new_tl):
        x = i - 30
        new_tl[i] = {**pt, "t": "now" if x == 0 else f"{x}s"}
    return new_tl


# ─── Derived metrics ─────────────────────────────────────────────────────────
def compute_occupancy_metrics(objects: List["TrackedObject"]) -> List[Tuple[str, int, str]]:
    """Return (label, percent_int, level) rows computed from live objects."""
    n_front = max(len([o for o in objects if o.bev_y > 0]), 1)

    in_path  = [o for o in objects if abs(o.bev_x) < 2.2 and 0 < o.bev_y < 20]
    pct_path = min(100, len(in_path) * 25)
    lv_path  = "crit" if pct_path > 60 else "warn" if pct_path > 25 else "caut"

    in_lane  = [o for o in objects if abs(o.bev_x) < 3.5 and o.bev_y > 0]
    pct_lane = min(100, round(len(in_lane) / n_front * 100))
    lv_lane  = "crit" if pct_lane > 70 else "warn" if pct_lane > 40 else "caut"

    if in_lane:
        min_headway = min(o.bev_y / max(EGO_SPEED_MS, 0.1) for o in in_lane)
        pct_headway = min(100, max(0, round((4.0 - min_headway) / 3.0 * 100)))
    else:
        pct_headway = 0
    lv_headway = "crit" if pct_headway > 70 else "warn" if pct_headway > 35 else "caut" if pct_headway else "ok"

    stopping_distance = EGO_SPEED_MS * EGO_SPEED_MS / (2.0 * 5.5) + 2.0
    if in_lane:
        min_psd = min(o.bev_y / max(stopping_distance, 0.1) for o in in_lane)
        pct_psd = min(100, max(0, round((1.5 - min_psd) / 1.5 * 100)))
    else:
        pct_psd = 0
    lv_psd = "crit" if pct_psd > 70 else "warn" if pct_psd > 35 else "caut" if pct_psd else "ok"

    max_decel = max([o.req_decel for o in objects], default=0.0)
    pct_decel = min(100, max(0, round(max_decel / 6.0 * 100)))
    lv_decel = "crit" if max_decel >= 5.0 else "warn" if max_decel >= 3.0 else "caut" if max_decel >= 1.0 else "ok"

    max_closing = max([-o.rel_vel for o in objects if o.rel_vel < 0], default=0.0)
    pct_delta_v = min(100, max(0, round(max_closing / 10.0 * 100)))
    lv_delta_v = "crit" if max_closing >= 8.0 else "warn" if max_closing >= 4.0 else "caut" if max_closing >= 1.0 else "ok"

    crowd = crowding_ahead(objects)
    pct_crowd = int(crowd["score"])
    lv_crowd = "warn" if crowd["warning"] else "caut" if pct_crowd else "ok"

    frame_id = AVAILABLE_FRAMES[current_frame_index % TOTAL_FRAMES] if AVAILABLE_FRAMES else None
    weather = infer_weather_from_image(frame_id)
    pct_weather = int(weather.get("score", 0))
    lv_weather = str(weather.get("level", "ok"))

    return [
        ("Lane Occupancy",     pct_lane,  lv_lane),
        ("Time Headway",       pct_headway, lv_headway),
        ("Stopping Margin",    pct_psd, lv_psd),
        ("Brake Demand",       pct_decel, lv_decel),
        ("Closing Speed",      pct_delta_v, lv_delta_v),
        ("Weather Risk",       pct_weather, lv_weather),
        ("Crowd Ahead",        pct_crowd, lv_crowd),
    ]


METRIC_DESCRIPTIONS = {
    "Ego-path Occupancy": "Objects inside the narrow ego path: |lateral| < 2.2 m and 0-20 m ahead.",
    "Lane Occupancy": "Share of forward tracked objects inside the ego lane corridor: |lateral| < 3.5 m.",
    "Time Headway": "Short temporal gap to the nearest object in the ego lane, estimated from forward distance / ego speed.",
    "Stopping Margin": "PSD-style risk: forward clearance compared with estimated stopping distance; high means little stopping margin.",
    "Brake Demand": "Required deceleration to avoid conflict, derived from relative velocity and remaining distance.",
    "Closing Speed": "Largest radar-derived closing speed toward ego; higher means stronger approach severity.",
    "Weather Risk": "Strict camera-image weather gate using brightness, contrast, saturation, dark-area and glare evidence.",
    "Crowd Ahead": f"Warning-only density cue: {CROWD_AHEAD_WARN_COUNT}+ objects within {CROWD_AHEAD_FORWARD_M:.0f} m and +/-{CROWD_AHEAD_LATERAL_M:.0f} m.",
    "CROWD": f"Incoming crowd: objects clustered in the path ahead (within {CROWD_AHEAD_FORWARD_M:.0f} m and +/-{CROWD_AHEAD_LATERAL_M:.0f} m). Warns at {CROWD_AHEAD_WARN_COUNT}+.",
}


def metric_description(label: str) -> str:
    return METRIC_DESCRIPTIONS.get(label, "")


def _risk_score_from_category(risk: str) -> float:
    return {"CRITICAL": 4.0, "HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}.get(risk, 1.0)


def get_min_ttc(objects: List[TrackedObject]) -> float:
    finite_ttcs = [o.ttc for o in objects if math.isfinite(o.ttc)]
    return min(finite_ttcs) if finite_ttcs else 6.0


def get_closest(objects: List[TrackedObject]) -> TrackedObject:
    if not objects:
        # Return a safe dummy so callers never crash on empty scenes
        return TrackedObject(
            id="--", cls="None", source="--", dist=99.0, rel_vel=0.0,
            ttc=float("inf"), req_decel=0.0, occupancy=0.0,
            path_conflict=False, confidence=0.0, risk="LOW", risk_reason="—",
            bev_x=0.0, bev_y=0.0, cam_cx=-1, cam_cy=-1, cam_w=-1, cam_h=-1,
        )
    return min(objects, key=lambda o: o.dist)

def get_high_risk_count(objects: List[TrackedObject]) -> int:
    return sum(1 for o in objects if o.risk in ("HIGH", "CRITICAL"))

def get_overall_risk(objects: List[TrackedObject]) -> str:
    if any(o.risk == "CRITICAL" for o in objects):
        return "CRITICAL"
    if any(o.risk == "HIGH" for o in objects):
        return "HIGH"
    if any(o.risk == "MEDIUM" for o in objects):
        return "MEDIUM"
    return "LOW"

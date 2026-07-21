"""Pre-built frame bundles for smooth playback (queue + rolling lookahead).

Each bundle holds the heavy Plotly artifacts for one frame index: BEV, camera
overlays, timeline chart, and KPI gauges. Background workers fill a lookahead
window ahead of the live playhead so the main callback mostly dequeues ready
work instead of building figures on the hot path.
"""

from __future__ import annotations

import copy
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go

# Prefetch enough frames that 1x/2x dwell can absorb occasional slow builds.
LOOKAHEAD_FRAMES = 28
MAX_CACHED_BUNDLES = 72
WORKER_COUNT = 4
# Live playback keeps the original 3D radar plane (Mesh3d / Scatter3d).


@dataclass(frozen=True)
class FrameBundle:
    frame_index: int
    frame_id: str
    bev_figure: go.Figure
    cam_figure: go.Figure
    timeline_figure: go.Figure
    gauge_figures: Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]
    timeline_data: list


class PlaybackFrameCache:
  def __init__(self) -> None:
    self._bundles: Dict[int, FrameBundle] = {}
    self._queue: Deque[int] = deque()
    self._queued: Set[int] = set()
    self._in_flight: Set[int] = set()
    self._timelines: List[list] = []
    self._timeline_built_through = -1
    self._center_index = 0
    self._lock = threading.Lock()
    self._worker_count = 0

  def reset(self) -> None:
    with self._lock:
      self._bundles.clear()
      self._queue.clear()
      self._queued.clear()
      self._in_flight.clear()
      self._timelines = []
      self._timeline_built_through = -1
      self._center_index = 0

  def schedule_ahead(self, center_index: int, lookahead: int = LOOKAHEAD_FRAMES) -> None:
    from data import TOTAL_FRAMES
    total = TOTAL_FRAMES or 0
    if total <= 0:
      return
    center_index = int(center_index) % total
    self._center_index = center_index
    to_enqueue: list[int] = []
    with self._lock:
      for offset in range(max(0, lookahead) + 1):
        idx = (center_index + offset) % total
        if idx in self._bundles or idx in self._in_flight or idx in self._queued:
          continue
        self._queue.append(idx)
        self._queued.add(idx)
        to_enqueue.append(idx)
    if to_enqueue:
      self._ensure_workers()

  def get(self, frame_index: int) -> Optional[FrameBundle]:
    with self._lock:
      return self._bundles.get(int(frame_index))

  def build_sync(self, frame_index: int) -> FrameBundle:
    frame_index = int(frame_index)
    existing = self.get(frame_index)
    if existing is not None:
      return existing
    bundle = self._build_bundle(frame_index)
    with self._lock:
      self._bundles[frame_index] = bundle
      self._in_flight.discard(frame_index)
      self._queued.discard(frame_index)
      self._evict_far_from(self._center_index)
    return bundle

  def _timeline_for_index(self, frame_index: int) -> list:
    """Build timeline states incrementally up to ``frame_index`` (no full-sequence hitch)."""
    from data import (
      TOTAL_FRAMES,
      generate_timeline,
      load_objects_for_frame_index,
      update_timeline,
    )
    total = TOTAL_FRAMES or 0
    if total <= 0:
      return []
    frame_index = int(frame_index) % total

    while True:
      with self._lock:
        if self._timeline_built_through >= frame_index:
          return copy.deepcopy(self._timelines[frame_index])
        next_idx = self._timeline_built_through + 1
        prev = None if next_idx == 0 else self._timelines[next_idx - 1]

      # Load / update outside the lock so other workers can dequeue jobs.
      if next_idx == 0:
        tl = generate_timeline(load_objects_for_frame_index(0))
      else:
        tl = update_timeline(prev, load_objects_for_frame_index(next_idx))

      with self._lock:
        if self._timeline_built_through == next_idx - 1:
          self._timelines.append(tl)
          self._timeline_built_through = next_idx
        # else another worker already advanced — loop and re-check.

  def _build_bundle(self, frame_index: int) -> FrameBundle:
    from data import AVAILABLE_FRAMES, load_objects_for_frame_index
    from figures import build_timeline_figure, get_bev_figure, get_camera_figure

    frame_id = AVAILABLE_FRAMES[frame_index]
    objects = load_objects_for_frame_index(frame_index)
    timeline_data = self._timeline_for_index(frame_index)
    bev_figure = get_bev_figure(frame_id, objects, 0)
    # Embed camera JPEG so Plotly paints the image immediately (URL layout_images
    # load async and flash the dark plot background between frames).
    cam_figure = get_camera_figure(frame_id, objects, embed_image=True)
    timeline_figure = build_timeline_figure(timeline_data)
    from callbacks import build_gauge_figures
    gauge_figures = build_gauge_figures(objects, frame_index=frame_index)
    return FrameBundle(
      frame_index=frame_index,
      frame_id=frame_id,
      bev_figure=bev_figure,
      cam_figure=cam_figure,
      timeline_figure=timeline_figure,
      gauge_figures=gauge_figures,
      timeline_data=timeline_data,
    )

  def _evict_far_from(self, center_index: int) -> None:
    if len(self._bundles) <= MAX_CACHED_BUNDLES:
      return
    from data import TOTAL_FRAMES
    total = TOTAL_FRAMES or 1

    def ring_distance(a: int, b: int) -> int:
      diff = abs(a - b)
      return min(diff, total - diff)

    victims = sorted(
      self._bundles.keys(),
      key=lambda idx: ring_distance(idx, center_index),
      reverse=True,
    )
    for idx in victims:
      if len(self._bundles) <= MAX_CACHED_BUNDLES:
        break
      if idx in self._in_flight:
        continue
      self._bundles.pop(idx, None)

  def _ensure_workers(self) -> None:
    with self._lock:
      while self._worker_count < WORKER_COUNT:
        self._worker_count += 1
        threading.Thread(
          target=self._worker_loop,
          name=f"playback-bundle-worker-{self._worker_count}",
          daemon=True,
        ).start()

  def _pop_next_job(self) -> Optional[int]:
    with self._lock:
      while self._queue:
        idx = self._queue.popleft()
        self._queued.discard(idx)
        if idx in self._bundles:
          continue
        if idx in self._in_flight:
          continue
        self._in_flight.add(idx)
        return idx
      self._worker_count = max(0, self._worker_count - 1)
      return None

  def _worker_loop(self) -> None:
    while True:
      frame_index = self._pop_next_job()
      if frame_index is None:
        return
      try:
        bundle = self._build_bundle(frame_index)
        with self._lock:
          self._bundles[frame_index] = bundle
          self._in_flight.discard(frame_index)
          self._evict_far_from(self._center_index)
      except Exception:
        with self._lock:
          self._in_flight.discard(frame_index)
        traceback.print_exc()


_playback_cache: Optional[PlaybackFrameCache] = None


def get_playback_cache() -> PlaybackFrameCache:
  global _playback_cache
  if _playback_cache is None:
    _playback_cache = PlaybackFrameCache()
  return _playback_cache


def reset_playback_cache() -> None:
  get_playback_cache().reset()


def warm_playback_figure_cache_async(start_index: int = 0, limit: Optional[int] = None) -> None:
  """Queue pre-build jobs for frames starting at ``start_index``."""
  lookahead = LOOKAHEAD_FRAMES if limit is None else int(limit)
  get_playback_cache().schedule_ahead(int(start_index), lookahead=lookahead)

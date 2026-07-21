"""
callbacks.py — All Dash callbacks (backend logic).
Registers all @app.callback functions; called once from app.py.
"""

import math
import threading
import time
from datetime import datetime, timedelta
from dash import Input, Output, State, html, ctx, no_update
import data as data_module

# Serializes the main playback render. The interval timer is always on and the
# dev server is threaded, so without this a slow (cold-cache) frame lets several
# queued ticks run concurrently, each advancing the shared frame counter and
# finishing out of order — which the user sees as skipped/jumping frames.
_PLAYBACK_LOCK = threading.Lock()
# Monotonic pacing: a new frame is shown only after the previous render finished
# AND the selected speed's minimum dwell time has elapsed. The UI timer polls fast
# (100 ms); speed is expressed here, not by skipping interval ticks.
_PLAYBACK_PACE = {"last_shown_mono": 0.0}
_PLAYBACK_POLL_MS = 100
from data import (
    dicts_to_objects, objects_to_dicts,
    objects_for_playback_tick,
    get_min_ttc, get_closest, get_high_risk_count, get_overall_risk,
    compute_occupancy_metrics, risk_threshold_legend,
    load_objects_for_frame_index, get_ego_motion_avg_kmh,
    crowding_ahead, metric_description, infer_weather_from_image,
    RISK_COLORS, OBJECT_CLASS_COLORS,
    RISK_TTC_THRESHOLDS, RISK_DIST_THRESHOLDS,
)


def _occ_bar(label: str, val: int, lv: str) -> html.Div:
    col = {"crit": T["red"], "warn": T["amber"], "caut": T["yellow"]}.get(lv, T["green"])
    desc = metric_description(label)
    return html.Div(title=desc, style={"marginBottom": 9, "cursor":"help"}, children=[
        html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":3}, children=[
            html.Span(label, title=desc, style={"fontSize":11,"color":T["bright"]}),
            html.Span(f"{val}%", style={"fontSize":11,"color":col,"fontFamily":"Share Tech Mono,monospace"}),
        ]),
        html.Div(style={"height":4,"background":T["border"],"borderRadius":2}, children=[
            html.Div(style={"height":"100%","width":f"{val}%","background":col,
                           "borderRadius":2,"boxShadow":f"0 0 5px {col}60"}),
        ]),
    ])
from figures import build_gauge
from playback_cache import (
    get_playback_cache,
    reset_playback_cache,
    warm_playback_figure_cache_async,
    LOOKAHEAD_FRAMES,
)

# ─── Static event log entries ────────────────────────────────────────────────
_EVENTS = [
    {"offset_s": 8,  "type": "CRITICAL", "msg": "Pedestrian ID:05 — TTC=1.8 s — Immediate risk, path conflict active"},
    {"offset_s": 13, "type": "WARNING",  "msg": "Car ID:02 — Closing speed −5.1 m/s, required decel 4.8 m/s²"},
    {"offset_s": 19, "type": "INFO",     "msg": "Camera weather inference active - strict thresholds, per-frame image checked"},
    {"offset_s": 36, "type": "INFO",     "msg": "Per-frame calibration validated and loaded, twowheelersSafety fusion active"},
    {"offset_s": 41, "type": "INFO",     "msg": "Two-wheeler fusion mode active — all sensors nominal, 10 objects tracked"},
]

# ─── Design tokens (shared with layout.py) ───────────────────────────────────
T = dict(
    panel="#0b1525", border="#152436",
    cyan="#00e5ff", green="#00e676", amber="#ffab00",
    red="#ff1744",  yellow="#ffd600",
    dim="#7793ae",  bright="#dceeff",
)


def _weather_icon(weather: dict) -> str:
    """Pick a glyph matching the inferred condition (handles night+rain etc.)."""
    if weather.get("label") == "Unknown":
        return "❓"
    is_night = bool(weather.get("is_night"))
    if weather.get("is_rain"):
        return "🌧"          # rain glyph reads at night or day
    if weather.get("is_fog"):
        return "🌫"
    if weather.get("is_glare"):
        return "🌤"
    if is_night:
        return "🌙"
    return "☀"


def _weather_status_el(frame_id: str) -> html.Span:
    weather = infer_weather_from_image(frame_id)
    level = str(weather.get("level", "ok"))
    col = {"crit": T["red"], "warn": T["amber"], "caut": T["yellow"]}.get(level, T["green"])
    label = str(weather.get("label", "Unknown"))
    score = int(weather.get("score", 0))
    detail = str(weather.get("detail", ""))
    icon = _weather_icon(weather)
    children = [
        html.Span(icon, style={"fontSize": "19px", "lineHeight": "1"}),
        html.Span(label, style={"fontSize": "15px", "color": col, "fontWeight": 700}),
    ]
    if score > 0:
        children.append(html.Span(f"risk {score}%", style={"fontSize": "12px", "color": T["dim"]}))
    return html.Span(
        title=detail,
        style={
            "display": "inline-flex", "alignItems": "center", "gap": "7px",
            "padding": "4px 12px", "borderRadius": "999px",
            "background": f"{col}1c", "border": f"1px solid {col}55",
            "fontFamily": "Share Tech Mono,monospace", "cursor": "help",
        },
        children=children,
    )


# ─── Helper: build the object risk HTML table ─────────────────────────────────
def _object_table(objects):
    RISK_ORDER = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
    rows_sorted = sorted(objects, key=lambda o: RISK_ORDER.get(o.risk, 4))

    HEADERS = ["ID","CLASS","DIST","REL.V","TTC (s)",
               "DECEL","OCCUP.","CF","CONF","DIRECTION","TRIGGER","RISK"]

    def _th(text):
        return html.Th(text, style={
            "padding":"6px 5px","textAlign":"left","fontSize":10,
            "color":T["dim"],"letterSpacing":".5px",
            "borderBottom":f"1px solid {T['border']}",
            "fontWeight":600,"whiteSpace":"nowrap",
        })

    def _occ_cell(val):
        col = "#ff1744" if val>.6 else "#ffab00" if val>.3 else "#00e676"
        return html.Td(style={"padding":"5px 4px"}, children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":3}, children=[
                html.Div(style={"width":30,"height":3,"background":T["border"],"borderRadius":1}, children=[
                    html.Div(style={"width":f"{val*100:.0f}%","height":"100%","background":col}),
                ]),
                html.Span(f"{val*100:.0f}%", style={"fontSize":8,"color":T["dim"]}),
            ]),
        ])

    row_els = []
    for o in rows_sorted:
        col     = OBJECT_CLASS_COLORS.get(o.cls, "#64748b")
        risk_col = RISK_COLORS.get(o.risk, "#00e676")
        ttc_finite = math.isfinite(o.ttc)
        ttc_str = f"{o.ttc:.1f}" if ttc_finite else "∞"
        # "∞" shown in cyan-dim to convey "safe / not approaching"
        ttc_col = (T["cyan"] if not ttc_finite
                   else ("#ff1744" if o.ttc < RISK_TTC_THRESHOLDS["CRITICAL"]
                         else "#ffab00" if o.ttc < RISK_TTC_THRESHOLDS["HIGH"] else T["bright"]))
        dst_col = ("#ff1744" if o.dist < RISK_DIST_THRESHOLDS["CRITICAL"]
                   else "#ffab00" if o.dist < RISK_DIST_THRESHOLDS["HIGH"]
                   else "#ffd600" if o.dist < RISK_DIST_THRESHOLDS["MEDIUM"] else T["bright"])

        row_els.append(html.Tr(
            style={"borderBottom":f"1px solid rgba(21,36,54,0.35)","cursor":"default"},
            children=[
                html.Td(o.id,  style={"padding":"6px 5px","fontFamily":"Share Tech Mono,monospace","color":T["bright"],"fontWeight":700,"fontSize":11}),
                html.Td(o.cls, style={"padding":"6px 5px","color":col,"fontSize":11,"fontWeight":600}),
                html.Td(f"{o.dist:.1f}",     style={"padding":"6px 5px","fontFamily":"Share Tech Mono,monospace","fontSize":11,"color":dst_col}),
                html.Td(f"{o.rel_vel:.1f}",  style={"padding":"6px 5px","fontFamily":"Share Tech Mono,monospace","fontSize":11,
                                                     "color":"#ff1744" if o.rel_vel<0 else "#00e676"}),
                html.Td(ttc_str,             style={"padding":"6px 5px","fontFamily":"Orbitron,monospace",
                                                     "fontSize":12,"fontWeight":700,"color":ttc_col}),
                html.Td(f"{o.req_decel:.1f}",style={"padding":"6px 5px","fontFamily":"Share Tech Mono,monospace","fontSize":11,"color":T["dim"]}),
                _occ_cell(o.occupancy),
                html.Td("⚠" if o.path_conflict else "—",
                        style={"padding":"6px 5px","textAlign":"center",
                               "color":"#ff1744" if o.path_conflict else T["dim"],"fontSize":11}),
                html.Td(f"{o.confidence:.2f}", style={"padding":"6px 5px","fontFamily":"Share Tech Mono,monospace","fontSize":11,
                    "color":"#00e676" if o.confidence>.8 else "#ffab00" if o.confidence>.5 else "#ff1744"}),
                html.Td(o.heading, style={"padding":"6px 5px","fontSize":11,
                    "color":T["bright"],"whiteSpace":"nowrap"}),
                html.Td(o.risk_reason, style={"padding":"6px 5px","fontSize":10,
                    "color":T["dim"],"maxWidth":120,"whiteSpace":"normal","lineHeight":1.2}),
                html.Td(style={"padding":"6px 5px"}, children=[
                    html.Span(o.risk, style={"fontFamily":"Orbitron,monospace","fontSize":9,"fontWeight":700,
                        "padding":"2px 6px","borderRadius":2,"color":risk_col,
                        "background":f"{risk_col}18","border":f"0.5px solid {risk_col}65"}),
                ]),
            ]))

    return html.Table(
        style={"width":"100%","borderCollapse":"collapse","fontSize":10},
        children=[
            html.Thead(html.Tr([_th(h) for h in HEADERS])),
            html.Tbody(row_els),
        ])


# ─── Helper: event log HTML ──────────────────────────────────────────────────
def _scene_time_str(offset_seconds: int = 0) -> str:
    """Match the status-bar clock (scene day/night anchor, real-time tick)."""
    scene_now = data_module.get_scene_clock(datetime.now())
    if offset_seconds:
        scene_now = scene_now - timedelta(seconds=int(offset_seconds))
    return scene_now.strftime("%H:%M:%S")


def _event_log(n_intervals, objects=None, scene_time=None):
    ev_col  = {"CRITICAL":T["red"],  "WARNING":T["amber"],  "INFO":T["cyan"]}
    ev_icon = {"CRITICAL":"⛔", "WARNING":"⚠", "INFO":"ℹ"}
    ev_bg   = {"CRITICAL":"rgba(255,23,68,.08)","WARNING":"rgba(255,171,0,.06)","INFO":"rgba(0,229,255,.04)"}
    if scene_time is not None:
        now_str = scene_time.strftime("%H:%M:%S")
    else:
        now_str = _scene_time_str(0)
    frame_info = data_module.get_frame_info(data_module.current_frame_index)
    weather = infer_weather_from_image(frame_info["frame_id"])
    weather_event = None
    if weather.get("level") in ("crit", "warn"):
        ev_type = "CRITICAL" if weather.get("level") == "crit" else "WARNING"
        weather_event = {
            "time": now_str,
            "type": ev_type,
            "msg": (
                f"Risky camera weather: {weather.get('label')} "
                f"({int(weather.get('score', 0))}%) - increase following margin."
            ),
        }
    crowd = crowding_ahead(objects or [])
    crowded_event = None
    if crowd["warning"]:
        nearest = crowd["nearest"]
        near_txt = f", nearest at {nearest:.1f} m" if math.isfinite(nearest) else ""
        crowded_event = {
            "time": now_str,
            "type": "WARNING",
            "msg": f"Incoming crowded area ahead - {crowd['count']} objects within 45 m / +/-12 m{near_txt}. Slow and monitor.",
        }

    # Rotate a dynamic entry into slot 0 every few ticks
    dynamic = [
        {"time": now_str,"type":"WARNING",
         "msg": "Pedestrian ID:05 — TTC updated, path conflict persists — reduce speed"},
        {"time": now_str,"type":"INFO",
         "msg": "Radar detections nominal - twowheelersSafety fusion locked"},
        {"time": now_str,"type":"CRITICAL",
         "msg": "Car ID:02 — distance margin critical, immediate braking required"},
    ]
    lead = weather_event or crowded_event or dynamic[n_intervals % len(dynamic)]
    if scene_time is not None:
        events = [lead] + [
            {
                "time": (scene_time - timedelta(seconds=ev["offset_s"])).strftime("%H:%M:%S"),
                "type": ev["type"],
                "msg": ev["msg"],
            }
            for ev in _EVENTS[:4]
        ]
    else:
        events = [lead] + [
            {
                "time": _scene_time_str(ev["offset_s"]),
                "type": ev["type"],
                "msg": ev["msg"],
            }
            for ev in _EVENTS[:4]
        ]

    items = []
    for ev in events:
        col = ev_col.get(ev["type"], T["cyan"])
        items.append(html.Div(style={
            "padding":"5px 7px","marginBottom":4,"borderRadius":2,
            "background":ev_bg.get(ev["type"],"rgba(0,229,255,.04)"),
            "borderLeft":f"2px solid {col}",
        }, children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":2}, children=[
                html.Span(f"{ev_icon.get(ev['type'],'')} {ev['type']}",
                    style={"fontSize":8,"fontFamily":"Orbitron,monospace","fontWeight":700,"color":col}),
                html.Span(ev["time"],
                    style={"fontSize":8,"color":T["dim"],"fontFamily":"Share Tech Mono,monospace"}),
            ]),
            html.Div(ev["msg"], style={"fontSize":9,"color":T["bright"],"lineHeight":1.4}),
        ]))
    return items


# ─── Helper: risk summary HTML ────────────────────────────────────────────────
def _risk_summary(objects, risk_str, risk_col, frame_id=None):
    counts = {
        "Critical": sum(1 for o in objects if o.risk == "CRITICAL"),
        "High":     sum(1 for o in objects if o.risk == "HIGH"),
        "Medium":   sum(1 for o in objects if o.risk == "MEDIUM"),
        "Low":      sum(1 for o in objects if o.risk == "LOW"),
    }
    colors = {"Critical":T["red"],"High":T["amber"],"Medium":T["yellow"],"Low":T["green"]}
    n = len(objects) or 1
    try:
        radar_count = str(int(data_module.load_radar_points(frame_id).shape[0])) if frame_id else "-"
    except Exception:
        radar_count = "-"

    bars = []
    for label, count in counts.items():
        col = colors[label]
        bars.append(html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":7}, children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":6}, children=[
                html.Div(style={"width":8,"height":8,"background":col,"borderRadius":1}),
                html.Span(label, style={"fontSize":9,"color":T["dim"]}),
            ]),
            html.Div(style={"display":"flex","alignItems":"center","gap":5}, children=[
                html.Div(style={"width":55,"height":3,"background":T["border"],"borderRadius":1}, children=[
                    html.Div(style={"width":f"{count/n*100:.0f}%","height":"100%","background":col}),
                ]),
                html.Span(str(count), style={"fontFamily":"Orbitron,monospace","fontSize":12,
                                             "color":col,"fontWeight":700,"minWidth":14,"textAlign":"right"}),
            ]),
        ]))

    threshold_rows = []
    for level, desc in risk_threshold_legend():
        threshold_rows.append(html.Div(style={"marginBottom":3}, children=[
            html.Span(level, style={"fontFamily":"Orbitron,monospace","fontSize":7,"fontWeight":700,
                                    "color":RISK_COLORS.get(level, T["dim"]),"marginRight":6}),
            html.Span(desc, style={"fontSize":8,"color":T["dim"],"lineHeight":1.25}),
        ]))

    return [
        html.Div(style={"textAlign":"center","padding":"8px 4px","marginBottom":10,
                        "background":f"{risk_col}10","border":f"1px solid {risk_col}42","borderRadius":3}, children=[
            html.Div("SCENE STATUS", style={"fontFamily":"Orbitron,monospace","fontSize":8,
                                            "color":T["dim"],"letterSpacing":"1px","marginBottom":3}),
            html.Div(risk_str, style={"fontFamily":"Orbitron,monospace","fontSize":15,"fontWeight":900,"color":risk_col}),
        ]),
        *bars,
        html.Div(style={"borderTop":f"1px solid {T['border']}","paddingTop":8,"marginTop":4,"marginBottom":8}, children=[
            html.Div("THRESHOLD RULES", style={"fontFamily":"Orbitron,monospace","fontSize":8,
                                               "color":T["dim"],"letterSpacing":"1px","marginBottom":5}),
            *threshold_rows,
            html.Div("+1 level when object is in ego path", style={"fontSize":8,"color":T["dim"],"marginTop":4,"fontStyle":"italic"}),
        ]),
        html.Div(style={"borderTop":f"1px solid {T['border']}","paddingTop":8,"marginTop":4}, children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":5}, children=[
                html.Span("Total Tracked", style={"fontSize":9,"color":T["dim"]}),
                html.Span(str(len(objects)), style={"fontFamily":"Orbitron,monospace","fontSize":13,"color":T["cyan"],"fontWeight":700}),
            ]),
            html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":5}, children=[
                html.Span("Ego Motion Avg",  style={"fontSize":9,"color":T["dim"]}),
                html.Span(f"{get_ego_motion_avg_kmh(frame_id):.0f} km/h radar est.",
                          style={"fontFamily":"Share Tech Mono,monospace","fontSize":9,"color":T["bright"]}),
            ]),
            html.Div(style={"display":"flex","justifyContent":"space-between"}, children=[
                html.Span("Radar Points", style={"fontSize":9,"color":T["dim"]}),
                html.Span(radar_count,     style={"fontFamily":"Share Tech Mono,monospace","fontSize":9,"color":T["bright"]}),
            ]),
        ]),
    ]


def build_header_widgets(objects, risk_str, now, *, scene_time=None):
  """Build status-bar HTML widgets shared by layout and live callback."""
  min_ttc   = get_min_ttc(objects)
  closest   = get_closest(objects)
  high_risk = get_high_risk_count(objects)
  RC = {"CRITICAL":T["red"],"HIGH":T["amber"],"MEDIUM":T["yellow"],"LOW":T["green"]}
  risk_col  = RC.get(risk_str, T["green"])
  risk_icon = {"CRITICAL":"⛔ ","HIGH":"⚠ ","MEDIUM":"⚡ ","LOW":"✓ "}.get(risk_str, "")

  # Only the risk-tinted top border is dynamic; grid/columns/padding are owned by
  # the .status-bar CSS class so the layout stays consistent (and never wraps).
  sb_style = {"borderTop": f"2px solid {risk_col}"}

  anim_cls = ("pulse-r" if risk_str=="CRITICAL" else "pulse-a" if risk_str=="HIGH" else "")
  _badge_cfg = {
      "CRITICAL": dict(fs=20, bdr=f"2px solid {risk_col}cc", bg=f"{risk_col}28", pad="6px 18px"),
      "HIGH":     dict(fs=18, bdr=f"2px solid {risk_col}99", bg=f"{risk_col}1a", pad="6px 16px"),
      "MEDIUM":   dict(fs=17, bdr=f"1px solid {risk_col}66", bg=f"{risk_col}12", pad="5px 15px"),
      "LOW":      dict(fs=16, bdr=f"1px solid {risk_col}44", bg=f"{risk_col}0c", pad="5px 14px"),
  }.get(risk_str, dict(fs=16, bdr=f"1px solid {risk_col}55", bg=f"{risk_col}15", pad="5px 14px"))
  _sub = {"CRITICAL":"Collision imminent — TTC or distance critical",
          "HIGH":"High risk — reduce speed",
          "MEDIUM":"Moderate risk — monitor closely",
          "LOW":"All metrics within safe bounds"}.get(risk_str,"")
  risk_badge = html.Div([
      html.Div(f"{risk_icon}{risk_str}", className=anim_cls, style={
          "fontFamily":"Orbitron,monospace","fontWeight":900,
          "fontSize":_badge_cfg["fs"],"color":risk_col,
          "padding":_badge_cfg["pad"],"border":_badge_cfg["bdr"],
          "borderRadius":3,"background":_badge_cfg["bg"],"display":"inline-block",
      }),
      html.Div(_sub, style={"fontSize":11,"color":risk_col,"opacity":"0.85",
                            "fontFamily":"Share Tech Mono,monospace","marginTop":4}),
  ], style={"textAlign":"center"})

  ttc_col  = (T["red"] if min_ttc < RISK_TTC_THRESHOLDS["CRITICAL"]
              else T["amber"] if min_ttc < RISK_TTC_THRESHOLDS["HIGH"] else T["cyan"])
  ttc_disp = f"{min_ttc:.1f}" if min_ttc < RISK_TTC_THRESHOLDS["MEDIUM"] else "∞"
  ttc_el = html.Div(ttc_disp, style={
      "fontFamily":"Orbitron,monospace","fontSize":22,"fontWeight":700,
      "lineHeight":"1","color":ttc_col,
  })

  cl_text = html.Div(f"{closest.cls}",
      style={"fontFamily":"Share Tech Mono,monospace","fontSize":13,"color":T["bright"]})
  cl_dist = html.Div([f"{closest.dist:.1f}", html.Span(" m", style={"fontSize":13})],
      style={"fontFamily":"Orbitron,monospace","fontSize":22,"color":T["cyan"],"fontWeight":700})

  hr_el = html.Div(str(high_risk), style={
      "fontFamily":"Orbitron,monospace","fontSize":30,"fontWeight":900,"lineHeight":"1",
      "color": T["red"] if high_risk > 0 else T["green"],
  })

  frame_info = data_module.get_frame_info(data_module.current_frame_index)
  ego_kmh = get_ego_motion_avg_kmh(frame_info["frame_id"])
  ego_val_el = f"{ego_kmh:.0f}"
  ego_src_el = "km/h radar est."
  frame_num_el = html.Div(
      frame_info['frame_id'],
      style={"fontFamily":"Orbitron,monospace","fontSize":26,"fontWeight":900,"color":T["cyan"],"lineHeight":"1"},
  )
  frame_calib_el = html.Div(
      f"{frame_info['index']}/{frame_info['total']} · {frame_info['calib']}",
      style={"fontSize":8,"color":T["dim"],"fontFamily":"Share Tech Mono,monospace","marginTop":2},
  )
  weather_el = _weather_status_el(frame_info["frame_id"])

  # Clock reflects the scene's day/night lighting (from the first frame) while
  # still ticking in real time; the date stays the real calendar date.
  scene_now = scene_time if scene_time is not None else data_module.get_scene_clock(now)
  date_ref = scene_now if scene_time is not None else now

  return (
      sb_style, risk_badge, ttc_el, cl_text, cl_dist, hr_el,
      ego_val_el, ego_src_el,
      frame_num_el, frame_calib_el,
      scene_now.strftime("%H:%M:%S"), date_ref.strftime("%d/%m/%Y"), weather_el, risk_col,
  )


def build_gauge_figures(objects, frame_index=None):
  min_ttc = get_min_ttc(objects)
  idx = data_module.current_frame_index if frame_index is None else int(frame_index)
  frame_info = data_module.get_frame_info(idx)
  ego_ms = max(get_ego_motion_avg_kmh(frame_info["frame_id"]) / 3.6, 0.1)
  lane_objects = [o for o in objects if abs(o.bev_x) < 3.5 and o.bev_y > 0]
  min_thw = min((o.bev_y / ego_ms for o in lane_objects), default=4.0)
  max_drac = max((o.req_decel for o in objects), default=0.0)
  # Stopping distance for the ego rider (v²/2a + reaction buffer).
  stopping_distance = ego_ms * ego_ms / (2.0 * 5.5) + 2.0
  # STOP MARGIN in metres: how much clear road remains beyond what the rider
  # needs to stop. Positive = safe buffer; near-zero/negative = danger. Using
  # metres (not a bare ratio) makes the unit unambiguous.
  closest_gap = min((o.bev_y for o in lane_objects), default=stopping_distance + 40.0)
  stop_margin_m = closest_gap - stopping_distance
  min_ttc_display = min(min_ttc, 6.0) if math.isfinite(min_ttc) else 6.0
  min_thw_display = min(min_thw, 4.0)
  max_drac_display = min(max_drac, 6.0)
  stop_margin_display = max(0.0, min(stop_margin_m, 40.0))
  g_lv = ("ok" if min_ttc >= RISK_TTC_THRESHOLDS["MEDIUM"]
          else "crit" if min_ttc < RISK_TTC_THRESHOLDS["CRITICAL"]
          else "warn" if min_ttc < RISK_TTC_THRESHOLDS["HIGH"] else "caut")
  thw_lv = "ok" if min_thw >= 2.5 else "crit" if min_thw < 1.0 else "warn" if min_thw < 1.8 else "caut"
  drac_lv = "ok" if max_drac < 1.0 else "crit" if max_drac >= 5.0 else "warn" if max_drac >= 3.0 else "caut"
  sm_lv = ("crit" if stop_margin_m < 2.0 else "warn" if stop_margin_m < 6.0
           else "caut" if stop_margin_m < 12.0 else "ok")

  # Incoming-crowd KPI: number of objects clustered in the path ahead. It is
  # highlighted as a warning whenever a crowd is present.
  crowd = crowding_ahead(objects)
  crowd_count = int(crowd["count"])
  crowd_max = 8
  crowd_lv = "warn" if crowd["warning"] else "caut" if crowd_count > 0 else "ok"

  return (
      build_gauge(min_ttc_display, 6, "s", g_lv, "MIN TTC", "Time to collision"),
      build_gauge(min_thw_display, 4, "s", thw_lv, "HEADWAY", "Time gap ahead"),
      build_gauge(max_drac_display, 6, "m/s²", drac_lv, "REQ DECEL", "Brake to avoid",
                  compact_number=True),
      build_gauge(stop_margin_display, 40, "m", sm_lv, "STOP MARGIN",
                  "Clear road beyond braking", valueformat=".0f"),
      build_gauge(min(crowd_count, crowd_max), crowd_max, "obj", crowd_lv,
                  "CROWD", "Objects ahead", valueformat=".0f"),
  )


_ALARM_BTN_BASE = {
    "fontFamily":"Share Tech Mono,monospace","fontSize":10,"color":"#ffffff",
    "border":"none","borderRadius":5,"padding":"6px 12px",
    "cursor":"pointer","transition":"background 120ms ease",
}

# Table / event log / risk summary — refresh less often than the camera/radar.
_SECONDARY_PANEL_REFRESH_TICKS = 8
# Gauges + timeline + occupancy are lighter than 3D but still costly to ship;
# refresh every few polls while still feeling live.
_GAUGE_PANEL_REFRESH_TICKS = 3


def _playback_pace_ms(speed_value) -> int:
    """Minimum milliseconds between successive frames for the selected speed."""
    speed = str(speed_value or "1x").lower()
    if "4" in speed:
        return 140
    if "2" in speed:
        return 260
    if "slow" in speed:
        return 900
    return 380


def _reset_playback_pace() -> None:
    _PLAYBACK_PACE["last_shown_mono"] = 0.0


def _playback_ready_for_next_frame(playback_speed) -> bool:
    """True when enough time has passed since the last completed frame render."""
    last = _PLAYBACK_PACE["last_shown_mono"]
    if last <= 0.0:
        return True
    elapsed_ms = (time.monotonic() - last) * 1000.0
    return elapsed_ms >= _playback_pace_ms(playback_speed)

# ─── Main registration function ───────────────────────────────────────────────
def register_callbacks(app):

    # ── Risk-differentiated alarm — beeps + TTS, utterance created inside setTimeout ──
    app.clientside_callback(
        """
        function(n_intervals, objs, enabled) {
            if (!enabled || !enabled.enabled) return '';
            if (!objs || !objs.length) return '';

            // Find highest-risk closest object
            var L = {CRITICAL:4, HIGH:3, MEDIUM:2, LOW:1};
            var best = null, bLv = 0;
            for (var i = 0; i < objs.length; i++) {
                var lv = L[objs[i].risk] || 1;
                if (lv > bLv || (lv === bLv && best && objs[i].dist < best.dist)) {
                    bLv = lv; best = objs[i];
                }
            }
            if (!best || bLv < 2) return '';

            // Per-level cooldown: CRITICAL 6 s, HIGH 10 s, MEDIUM 18 s
            var cd = {4:6000, 3:10000, 2:18000}[bLv] || 10000;
            var now = Date.now();
            if (window._alT && (now - window._alT) < cd) return '';
            window._alT = now;

            // Beeps: CRITICAL=3 square, HIGH=2 sine, MEDIUM=1 soft sine
            try {
                var AC = window.AudioContext || window.webkitAudioContext;
                if (AC) {
                    var ctx = new AC();
                    var BP = {4:[0,.25,.50], 3:[0,.38], 2:[0]};
                    var BF = {4:960, 3:660, 2:440};
                    var BT = {4:'square', 3:'sine', 2:'sine'};
                    var BV = {4:.38, 3:.28, 2:.18};
                    BP[bLv].forEach(function(t) {
                        var o = ctx.createOscillator(), g = ctx.createGain();
                        o.connect(g); g.connect(ctx.destination);
                        o.type = BT[bLv]; o.frequency.value = BF[bLv];
                        g.gain.setValueAtTime(BV[bLv], ctx.currentTime + t);
                        g.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + t + .25);
                        o.start(ctx.currentTime + t);
                        o.stop(ctx.currentTime  + t + .28);
                    });
                }
            } catch(e) {}

            // Build speech text as a plain string (captured by closure, immune to GC)
            var dm  = (Math.round(best.dist * 10) / 10).toString();
            var tc  = (best.ttc && best.ttc < 9999) ? best.ttc.toFixed(1) + ' seconds' : null;
            var msg = bLv === 4
                ? ('Critical alert! There is a ' + best.cls + ' at ' + dm + ' meters.' +
                   (tc ? ' Time to collision ' + tc + '.' : '') +
                   ' Immediate action required.')
                : bLv === 3
                ? ('Warning! There is a ' + best.cls + ' at ' + dm +
                   ' meters. High risk. Reduce speed.')
                : ('Caution. ' + best.cls + ' detected at ' + dm + ' meters.');
            var rate = bLv === 4 ? 1.1 : 1.0;

            // Create SpeechSynthesisUtterance INSIDE setTimeout so it is alive when speak() runs.
            if (window.speechSynthesis) {
                var delay = bLv === 4 ? 700 : bLv === 3 ? 420 : 200;
                setTimeout(function() {
                    try {
                        window.speechSynthesis.cancel();
                        window._spk = new SpeechSynthesisUtterance(msg);
                        window._spk.lang   = 'en-US';
                        window._spk.rate   = rate;
                        window._spk.volume = 1.0;
                        window.speechSynthesis.speak(window._spk);
                    } catch(e) {}
                }, delay);
            }
            return '';
        }
        """,
        Output("_alarm_sink", "children"),
        Input("main-interval", "n_intervals"),
        [State("objects-store", "data"), State("alarm-enabled", "data")],
        prevent_initial_call=True,
    )

    # ── Start monitoring ─────────────────────────────────────────────────────
    @app.callback(
        [Output("audio-unlock-overlay", "style"),
         Output("monitoring-active", "data"),
         Output("alarm-enabled", "data"),
         Output("alarm-toggle-btn", "children"),
         Output("alarm-toggle-btn", "style"),
         Output("objects-store", "data", allow_duplicate=True),
         Output("frame-index-store", "data", allow_duplicate=True)],
        Input("audio-unlock-btn", "n_clicks"),
        State("startup-alarm-check", "value"),
        prevent_initial_call=True,
    )
    def start_monitoring(_n_clicks, alarm_values):
        alarm_on = "alarm" in (alarm_values or [])
        label = "🔊 Alarm ON" if alarm_on else "🔕 Alarm OFF"
        btn_style = {
            **_ALARM_BTN_BASE,
            "background": T["green"] if alarm_on else T["dim"],
            "boxShadow": "0 0 8px rgba(0,230,118,0.24)" if alarm_on else "none",
        }
        data_module.current_frame_index = 0
        data_module.playback_tick = 0
        _reset_playback_pace()
        reset_playback_cache()
        init_objects = load_objects_for_frame_index(0)
        # Build frame 0 synchronously so START MONITORING never opens on a black
        # unfinished plot, then prefetch the next window in the background.
        get_playback_cache().build_sync(0)
        warm_playback_figure_cache_async(0)
        return (
            {"display": "none"},
            {"active": True},
            {"enabled": alarm_on},
            label,
            btn_style,
            objects_to_dicts(init_objects),
            0,
        )

    app.clientside_callback(
        """
        function(n) {
            if (!n) return window.dash_clientside.no_update;
            try {
                var AC = window.AudioContext || window.webkitAudioContext;
                if (AC) { var ctx = new AC(); ctx.resume(); }
            } catch(e) {}
            try {
                if (window.speechSynthesis) {
                    var u = new SpeechSynthesisUtterance('');
                    u.volume = 0;
                    window.speechSynthesis.speak(u);
                }
            } catch(e) {}
            return '';
        }
        """,
        Output("_alarm_sink", "children", allow_duplicate=True),
        Input("audio-unlock-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    # ── Alarm toggle button ────────────────────────────────────────────────────
    @app.callback(
        [Output("alarm-enabled",    "data"),
         Output("alarm-toggle-btn", "children"),
         Output("alarm-toggle-btn", "style")],
        Input("alarm-toggle-btn", "n_clicks"),
        State("alarm-enabled",      "data"),
        prevent_initial_call=True,
    )
    def toggle_alarm_sound(n_clicks, enabled_data):
        enabled = not (enabled_data or {}).get("enabled", False)
        label   = "🔊 Alarm ON" if enabled else "🔕 Alarm OFF"
        style   = {**_ALARM_BTN_BASE,
                   "background": T["green"] if enabled else T["dim"],
                   "boxShadow":  "0 0 8px rgba(0,230,118,0.24)" if enabled else "none"}
        return {"enabled": enabled}, label, style


    @app.callback(
        [
            Output("pause-store",        "data"),
            Output("pause-button",       "children"),
            Output("pause-button",       "style"),
        ],
        Input("pause-button", "n_clicks"),
        [State("pause-store", "data"), State("monitoring-active", "data")],
        prevent_initial_call=True,
    )
    def toggle_pause(n_clicks, pause_data, monitoring):
        paused = pause_data.get("paused", False) if pause_data else False
        paused = not paused
        monitoring_on = bool((monitoring or {}).get("active"))
        if paused:
            label = "▶ Resume"
            style = {
                "fontFamily":"Share Tech Mono,monospace","fontSize":10,"color":"#ffffff",
                "background":T["green"],"border":"none","borderRadius":5,"padding":"7px 14px",
                "cursor":"pointer","boxShadow":"0 0 12px rgba(0,230,118,0.24)",
                "transition":"background 120ms ease, transform 120ms ease",
            }
        else:
            label = "⏸ Pause"
            style = {
                "fontFamily":"Share Tech Mono,monospace","fontSize":10,"color":"#ffffff",
                "background":T["red"],"border":"none","borderRadius":5,"padding":"7px 14px",
                "cursor":"pointer","boxShadow":"0 0 12px rgba(255,23,68,0.24)",
                "transition":"background 120ms ease, transform 120ms ease",
            }
            if monitoring_on:
                _reset_playback_pace()
        return {"paused": paused}, label, style

    @app.callback(
        Output("main-interval", "interval"),
        Input("playback-speed", "value"),
        prevent_initial_call=True,
    )
    def configure_playback_interval(_playback_speed):
        """Keep a fast poll rate; playback speed is enforced by monotonic pacing."""
        return _PLAYBACK_POLL_MS

    @app.callback(
        [
            Output("objects-store",      "data"),
            Output("timeline-store",     "data"),
            Output("frame-index-store",   "data"),
            # Status bar
            Output("status-bar",         "style"),
            Output("overall-risk-badge", "children"),
            Output("min-ttc-value",      "children"),
            Output("closest-obj-text",   "children"),
            Output("closest-dist-value", "children"),
            Output("high-risk-count",    "children"),
            Output("ego-motion-value",   "children"),
            Output("ego-motion-source",  "children"),
            Output("frame-num-value",    "children"),
            Output("frame-calib-value",  "children"),
            Output("time-display",       "children"),
            Output("date-display",       "children"),
            Output("weather-status",     "children"),
            Output("preload-frame-image","src"),
            # Main figures
            Output("bev-graph",      "figure"),
            Output("cam-graph",      "figure"),
            Output("timeline-graph", "figure"),
            # Table + log
            Output("object-table", "children"),
            Output("tracked-count-badge", "children"),
            Output("event-log",    "children"),
            # Gauges
            Output("gauge-ttc",      "figure"),
            Output("gauge-conflict", "figure"),
            Output("gauge-uncert",   "figure"),
            Output("gauge-conf",     "figure"),
            Output("gauge-crowd",    "figure"),
            # Summary + occupancy
            Output("risk-summary",    "children"),
            Output("occupancy-bars",  "children"),
        ],
        Input("main-interval", "n_intervals"),
        Input("monitoring-active", "data"),
        [
            State("objects-store",    "data"),
            State("timeline-store",   "data"),
            State("pause-store",      "data"),
            State("frame-index-store","data"),
            State("playback-speed",   "value"),
        ],
        prevent_initial_call=True,
    )
    def update_all(n_intervals, monitoring, objects_data, timeline_data, pause, frame_index, playback_speed):
        now = datetime.now()
        monitoring_on = bool((monitoring or {}).get("active"))
        paused = bool((pause or {}).get("paused", False))
        trigger_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        total_frames = data_module.TOTAL_FRAMES or 1
        frame_index = int(frame_index or 0)
        tick = int(n_intervals or 0)
        interval_tick = trigger_prop.startswith("main-interval")
        # The timer ticks continuously. A tick only advances the sequence while
        # monitoring is active and not paused; otherwise we do nothing so the
        # splash screen and paused playback stay idle and cheap.
        advance_frame = monitoring_on and not paused and interval_tick
        if interval_tick and (not monitoring_on or paused):
            return (no_update,) * 30

        # Fast poll + monotonic pacing: never advance until the previous frame
        # finished rendering AND the selected speed's dwell time has elapsed.
        if advance_frame and not _playback_ready_for_next_frame(playback_speed):
            return (no_update,) * 30

        # Serialize the whole render. If a render is already in flight, skip this
        # poll — the next 100 ms tick will try again. Only one render runs at a
        # time, so the frame counter always moves 0,1,2,… with no gaps.
        if interval_tick:
            if not _PLAYBACK_LOCK.acquire(blocking=False):
                return (no_update,) * 30
        else:
            _PLAYBACK_LOCK.acquire()
        frame_shown = False
        try:
            # The frame counter is owned server-side (data_module.current_frame_index),
            # NOT read back from the client store. The timer fires faster than a store
            # round-trip completes, so a client-echoed index would always be stale and
            # the sequence would freeze at frame 0/1. Advancing server-side keeps
            # playback moving smoothly regardless of round-trip latency.
            if not monitoring_on or trigger_prop.startswith("monitoring-active"):
                frame_index = 0
                _reset_playback_pace()
            elif advance_frame:
                frame_index = (int(data_module.current_frame_index) + 1) % total_frames
            else:
                frame_index = int(data_module.current_frame_index)

            playback_cache = get_playback_cache()
            if monitoring_on:
                playback_cache.schedule_ahead(frame_index)

            bundle = playback_cache.get(frame_index)
            if bundle is None:
                # Never advance into a missing frame — that freezes the UI on a
                # sync 3D/camera build and flashes black panels. Hold the last
                # good frame until the prefetch worker catches up.
                if advance_frame:
                    playback_cache.schedule_ahead(frame_index, lookahead=LOOKAHEAD_FRAMES)
                    return (no_update,) * 30
                bundle = playback_cache.build_sync(frame_index)

            data_module.current_frame_index = frame_index
            data_module.playback_tick = frame_index

            if monitoring_on:
                objects = objects_for_playback_tick(frame_index)
            else:
                objects = dicts_to_objects(objects_data)

            risk_str = get_overall_risk(objects)
            timeline = timeline_data
            if advance_frame or trigger_prop.startswith("monitoring-active"):
                timeline = bundle.timeline_data

            (
                sb_style, risk_badge, ttc_el, cl_text, cl_dist, hr_el,
                ego_val_el, ego_src_el,
                frame_num_el, frame_calib_el,
                time_str, date_str, weather_el, risk_col,
            ) = build_header_widgets(objects, risk_str, now)

            current_frame_id = (
                data_module.AVAILABLE_FRAMES[frame_index]
                if data_module.AVAILABLE_FRAMES else "0000"
            )
            preload_frame_id = (
                data_module.AVAILABLE_FRAMES[(frame_index + 1) % total_frames]
                if data_module.AVAILABLE_FRAMES else "0000"
            )
            preload_src = f"/frame-image/{preload_frame_id}"
            bev_fig = bundle.bev_figure
            cam_fig = bundle.cam_figure
            tl_fig = bundle.timeline_figure
            g1, g2, g3, g4, g5 = bundle.gauge_figures
            tracked_badge = f"{len(objects)} TRACKED"

            gauge_due = (
                not interval_tick
                or tick % _GAUGE_PANEL_REFRESH_TICKS == 0
            )
            if gauge_due:
                tl_fig = bundle.timeline_figure
                occ_html = [_occ_bar(lbl, val, lv) for lbl, val, lv in compute_occupancy_metrics(objects)]
            else:
                tl_fig = no_update
                g1 = g2 = g3 = g4 = g5 = no_update
                occ_html = no_update

            secondary_due = (
                not interval_tick
                or tick % _SECONDARY_PANEL_REFRESH_TICKS == 0
            )
            if secondary_due:
                table_html   = _object_table(objects)
                events_html  = _event_log(n_intervals, objects)
                summary_html = _risk_summary(objects, risk_str, risk_col, current_frame_id)
            else:
                table_html = no_update
                events_html = no_update
                summary_html = no_update

            frame_shown = True
            return (
                objects_to_dicts(objects), timeline, frame_index,
                sb_style,
                risk_badge, ttc_el, cl_text, cl_dist, hr_el,
                ego_val_el, ego_src_el,
                frame_num_el, frame_calib_el,
                time_str, date_str, weather_el, preload_src,
                bev_fig, cam_fig, tl_fig,
                table_html, tracked_badge, events_html,
                g1, g2, g3, g4, g5,
                summary_html, occ_html,
            )
        finally:
            if frame_shown:
                _PLAYBACK_PACE["last_shown_mono"] = time.monotonic()
            _PLAYBACK_LOCK.release()

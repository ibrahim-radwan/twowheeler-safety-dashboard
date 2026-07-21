"""
layout.py — Full Dash HTML layout.
Imports from data.py and figures.py; no callbacks here.
"""

import base64
import math
from datetime import datetime
from dash import dcc, html
from data import (
    INITIAL_OBJECTS, AVAILABLE_FRAMES, TOTAL_FRAMES, generate_timeline,
    objects_to_dicts, compute_occupancy_metrics, get_overall_risk,
    metric_description,
)
from figures import (
    build_timeline_figure,
    get_bev_figure, get_camera_figure,
)
from callbacks import (
    build_header_widgets, build_gauge_figures,
    _object_table, _event_log, _risk_summary,
)

# ─── Design tokens (mirrors CSS variables) ───────────────────────────────────
T = dict(
    panel="#0b1525", border="#152436",
    cyan="#00e5ff", green="#00e676", amber="#ffab00",
    red="#ff1744", yellow="#ffd600",
    text="#b0cce0", dim="#7793ae", bright="#dceeff",
)

DEFAULT_PLAYBACK_INTERVAL_MS = 100

_GRAPH_CFG = {
    "displayModeBar": False,
    # Static plots skip Plotly interaction/event plumbing — fine for gauges /
    # timeline, but NOT for WebGL 3D radar (staticPlot blanks Mesh3d/Scatter3d).
    "staticPlot": True,
    "responsive": True,
    "scrollZoom": False,
}

# Camera + radar need the full Plotly renderer (layout images / WebGL 3D).
_GRAPH_CFG_VISUAL = {
    "displayModeBar": False,
    "staticPlot": False,
    "responsive": True,
    "scrollZoom": False,
}


def _logo_animated_svg(spin_s: float = 3.2) -> str:
    """Two-Wheeler Safety mark, re-drawn in the dashboard's own palette
    (cyan / medium-blue / navy / amber) so it blends with the dark UI instead
    of the white source artwork.

    Concept mirrors the brand logo: a **camera wheel** (left, lens + viewfinder
    brackets) and a **radar wheel** (right, scanning arcs + shield) joined by a
    frame bar. Both wheels' spokes rotate continuously (clockwise = rolling
    forward) via native SVG SMIL <animateTransform>, which runs even inside an
    <img> data-URI — giving a real cycling motion without a GIF.
    """
    CY = "#00e5ff"   # cyan  — primary dashboard accent
    LB = "#8becff"   # light cyan — highlights / brackets
    MB = "#2f6bff"   # medium blue — tyres
    NV = "#0a1b33"   # deep navy — hubs / lens body
    OR = "#ffab00"   # amber — safety shield

    lcx, rcx, cy, R = 60.0, 150.0, 56.0, 30.0

    def spokes(cx: float) -> str:
        hub, rim = R * 0.30, R * 0.82
        segs = []
        for k in range(8):
            a = math.radians(k * 45)
            x1, y1 = cx + hub * math.cos(a), cy + hub * math.sin(a)
            x2, y2 = cx + rim * math.cos(a), cy + rim * math.sin(a)
            segs.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{CY}" stroke-width="1.7" opacity="0.75"/>')
        return "".join(segs)

    def wheel(cx: float) -> str:
        rim = R * 0.82
        return (
            # static tyre (two rings)
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{MB}" stroke-width="4.6"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{R-3.6:.1f}" fill="none" stroke="{CY}" '
            f'stroke-width="1.1" opacity="0.55"/>'
            # rotating group: spokes + rim ring + a bright valve dot so the spin is visible
            f'<g>'
            f'{spokes(cx)}'
            f'<circle cx="{cx}" cy="{cy}" r="{rim:.1f}" fill="none" stroke="{CY}" '
            f'stroke-width="0.9" opacity="0.30"/>'
            f'<circle cx="{cx + rim - 2:.1f}" cy="{cy}" r="2.1" fill="{LB}"/>'
            f'<animateTransform attributeName="transform" type="rotate" '
            f'from="0 {cx} {cy}" to="360 {cx} {cy}" dur="{spin_s}s" repeatCount="indefinite"/>'
            f'</g>'
        )

    def brackets(cx: float) -> str:
        # camera-viewfinder corner brackets around the left wheel
        d, ln = R + 6, 8
        corners = [(cx - d, cy - d, 1, 1), (cx + d, cy - d, -1, 1),
                   (cx + d, cy + d, -1, -1), (cx - d, cy + d, 1, -1)]
        segs = []
        for x, y, sx, sy in corners:
            segs.append(f'<path d="M {x + sx*ln:.1f} {y:.1f} L {x:.1f} {y:.1f} '
                        f'L {x:.1f} {y + sy*ln:.1f}" fill="none" stroke="{LB}" '
                        f'stroke-width="2" stroke-linecap="round" opacity="0.9"/>')
        return "".join(segs)

    def camera_lens(cx: float) -> str:
        return (
            f'<circle cx="{cx}" cy="{cy}" r="10.5" fill="{NV}" stroke="{CY}" stroke-width="2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="6.2" fill="none" stroke="{MB}" stroke-width="2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="2.6" fill="{LB}"/>'
            f'<circle cx="{cx-3:.0f}" cy="{cy-3:.0f}" r="1.1" fill="#ffffff" opacity="0.85"/>'
        )

    def radar(cx: float) -> str:
        segs = [f'<circle cx="{cx}" cy="{cy}" r="4.6" fill="{NV}" stroke="{CY}" stroke-width="1.8"/>']
        for rr in (13.0, 20.0, 27.0):
            a0, a1 = math.radians(-68), math.radians(-12)
            ax, ay = cx + rr*math.cos(a0), cy + rr*math.sin(a0)
            bx, by = cx + rr*math.cos(a1), cy + rr*math.sin(a1)
            segs.append(f'<path d="M {ax:.1f} {ay:.1f} A {rr:.1f} {rr:.1f} 0 0 1 {bx:.1f} {by:.1f}" '
                        f'fill="none" stroke="{CY}" stroke-width="1.7" '
                        f'stroke-dasharray="3 3" opacity="0.85"/>')
        # sweeping scan arm (rotates within the top-right sector)
        arm_len = 27.0
        ex, ey = cx + arm_len*math.cos(math.radians(-40)), cy + arm_len*math.sin(math.radians(-40))
        segs.append(
            f'<g>'
            f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{LB}" stroke-width="2"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.2" fill="{LB}"/>'
            f'<animateTransform attributeName="transform" type="rotate" '
            f'values="-22 {cx} {cy}; 8 {cx} {cy}; -22 {cx} {cy}" '
            f'dur="{spin_s*1.1:.2f}s" repeatCount="indefinite"/>'
            f'</g>'
        )
        return "".join(segs)

    def shield() -> str:
        sx, sy = rcx + 26, cy + 20
        return (
            f'<path d="M {sx-7} {sy-8} L {sx+7} {sy-8} L {sx+7} {sy+1} '
            f'Q {sx+7} {sy+8} {sx} {sy+11} Q {sx-7} {sy+8} {sx-7} {sy+1} Z" '
            f'fill="{OR}" stroke="#0a1b33" stroke-width="1"/>'
            f'<path d="M {sx-3.2} {sy} L {sx-0.8} {sy+2.6} L {sx+3.6} {sy-3} " '
            f'fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    return (
        '<svg viewBox="0 0 210 100" xmlns="http://www.w3.org/2000/svg">'
        # frame bar joining the two hubs
        f'<line x1="{lcx+6}" y1="{cy}" x2="{rcx-6}" y2="{cy}" stroke="{MB}" '
        f'stroke-width="5" stroke-linecap="round"/>'
        f'<line x1="{lcx+6}" y1="{cy}" x2="{rcx-6}" y2="{cy}" stroke="{CY}" '
        f'stroke-width="1.6" stroke-linecap="round" opacity="0.7"/>'
        f'{wheel(lcx)}{wheel(rcx)}'
        f'{brackets(lcx)}{camera_lens(lcx)}'
        f'{radar(rcx)}{shield()}'
        '</svg>'
    )


def _logo_img(height: int = 44, radius: int = 8, glow: bool = True) -> html.Div:
    """Animated Two-Wheeler Safety mark themed to the dark dashboard.
    `height` sets the rendered height (icon is 2.1:1). `radius`/`glow` control
    the surrounding chip so it sits cleanly on the panel background."""
    svg = _logo_animated_svg()
    b64 = base64.b64encode(svg.encode()).decode()
    return html.Div(
        html.Img(src=f"data:image/svg+xml;base64,{b64}",
                 style={"height": f"{height}px", "width": "auto", "display": "block"}),
        style={
            "background": "linear-gradient(135deg,#0b1a30,#0a1424)",
            "borderRadius": f"{radius}px",
            "padding": "3px 8px",
            "border": "1px solid rgba(0,229,255,0.22)",
            "boxShadow": "0 0 14px rgba(0,229,255,0.28)" if glow else "none",
            "display": "inline-flex", "alignItems": "center",
        },
    )


# ─── Reusable helpers ────────────────────────────────────────────────────────
def panel_header(title: str, badge: str = None, badge_color: str = None,
                 badge_id: str = None) -> html.Div:
    bc = badge_color or T["dim"]
    right = []
    if badge:
        badge_props = {}
        if badge_id:
            badge_props["id"] = badge_id
        right.append(html.Span(badge, **badge_props, style={
            "fontFamily":"Share Tech Mono,monospace","fontSize":11,"color":bc,
            "border":f"0.5px solid {bc}55","padding":"1px 6px",
            "borderRadius":2,"background":f"{bc}12",
        }))
    return html.Div(className="panel-header", style={
        "padding":"5px 10px","borderBottom":f"1px solid {T['border']}",
        "display":"flex","alignItems":"center","justifyContent":"space-between",
        "background":"rgba(0,0,0,.28)",
    }, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"7px"}, children=[
            html.Div(style={"width":4,"height":4,"borderRadius":"50%","background":T["cyan"],"flexShrink":0}),
            html.Span(title, style={
                "fontFamily":"Orbitron,monospace","fontSize":12,"fontWeight":700,
                "letterSpacing":"1px","color":T["cyan"],"textTransform":"uppercase",
            }),
        ]),
        html.Div(children=right),
    ])


def panel(title: str, children: list, badge: str = None,
          badge_color: str = None, style: dict = None,
          className: str = None, badge_id: str = None) -> html.Div:
    if isinstance(title, str) and title.startswith("RADAR ROAD VIEW"):
        title = "RADAR 3D ROAD VIEW"
    s = {"background":T["panel"],"border":f"1px solid {T['border']}",
         "borderRadius":4,"overflow":"hidden","position":"relative"}
    if style:
        s.update(style)
    classes = "panel" if not className else f"panel {className}"
    return html.Div(
        className=classes,
        style=s,
        children=[panel_header(title, badge, badge_color, badge_id)] + list(children),
    )


def _gauge_cell(name: str, sub: str, graph_id: str, figure, *, extra_class: str = "") -> html.Div:
    """Top-bar KPI cell: static, always-legible title/subtitle above the gauge."""
    classes = "gauge-cell"
    if extra_class:
        classes = f"{classes} {extra_class}"
    return html.Div(className=classes, children=[
        html.Div(className="gauge-cell-head", children=[
            html.Div(name, className="gauge-cell-name",
                     title=metric_description(name)),
            html.Div(sub, className="gauge-cell-sub"),
        ]),
        dcc.Graph(id=graph_id, figure=figure, config=_GRAPH_CFG,
                  responsive=True, className="gauge-cell-graph"),
    ])


def occ_bar(label: str, val: int, lv: str) -> html.Div:
    col = {"crit":T["red"],"warn":T["amber"],"caut":T["yellow"]}.get(lv, T["green"])
    desc = metric_description(label)
    return html.Div(title=desc, style={"marginBottom":9, "cursor":"help"}, children=[
        html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":3}, children=[
            html.Span(label, title=desc, style={"fontSize":11,"color":T["text"]}),
            html.Span(f"{val}%", style={"fontSize":11,"color":col,"fontFamily":"Share Tech Mono,monospace"}),
        ]),
        html.Div(style={"height":4,"background":T["border"],"borderRadius":2}, children=[
            html.Div(style={"height":"100%","width":f"{val}%","background":col,
                           "borderRadius":2,"boxShadow":f"0 0 5px {col}60"}),
        ]),
    ])


# ─── Main layout factory ─────────────────────────────────────────────────────
def create_layout() -> html.Div:
    # Bootstrap initial state
    init_objs  = INITIAL_OBJECTS
    init_tl    = generate_timeline(init_objs)
    overall_r  = get_overall_risk(init_objs)
    now = datetime.now()
    (
        init_sb_style, init_risk_badge, init_ttc_el, init_cl_text, init_cl_dist,
        init_hr_el, init_ego_value, init_ego_source,
        init_frame_num, init_frame_calib,
        init_time, init_date, init_weather_el, init_risk_col,
    ) = build_header_widgets(init_objs, overall_r, now)
    g1, g2, g3, g4, g5 = build_gauge_figures(init_objs)
    init_table = _object_table(init_objs)
    init_events = _event_log(0, init_objs)
    init_frame = AVAILABLE_FRAMES[0] if AVAILABLE_FRAMES else "0000"
    preload_frame = AVAILABLE_FRAMES[1 % TOTAL_FRAMES] if TOTAL_FRAMES else init_frame
    init_summary = _risk_summary(init_objs, overall_r, init_risk_col, init_frame)
    bev_fig  = get_bev_figure(init_frame, init_objs, 0)
    cam_fig  = get_camera_figure(init_frame, init_objs, embed_image=True)
    tl_fig   = build_timeline_figure(init_tl)

    return html.Div(className="dashboard", children=[
        # ── Data stores & live interval ──────────────────────────────────────
        dcc.Store(id="objects-store",  data=objects_to_dicts(init_objs)),
        dcc.Store(id="timeline-store", data=init_tl),
        dcc.Store(id="pause-store",    data={"paused": False}),
        dcc.Store(id="monitoring-active", data={"active": False}),
        dcc.Store(id="alarm-enabled",  data={"enabled": False}),
        dcc.Store(id="frame-index-store", data=0),
        # Always-on timer: it ticks continuously from page load so playback can
        # never be left frozen by a mis-fired enable callback. Whether a tick
        # actually advances the sequence is decided server-side in update_all
        # (based on monitoring/pause state), and its rate is tuned per speed.
        dcc.Interval(id="main-interval", interval=DEFAULT_PLAYBACK_INTERVAL_MS, n_intervals=0, disabled=False),
        html.Div(id="_alarm_sink", style={"display":"none"}),
        html.Img(id="preload-frame-image", src=f"/frame-image/{preload_frame}", style={"display":"none"}),

        # ── Audio-unlock overlay (browser requires a user gesture before audio plays) ──
        html.Div(id="audio-unlock-overlay", children=[
            html.Div(style={
                "display":"flex","flexDirection":"column","alignItems":"center","gap":"16px",
            }, children=[
                html.Div(_logo_img(150, radius=16), style={
                    "filter":"drop-shadow(0 0 30px rgba(0,229,255,0.40))",
                    "display":"flex","justifyContent":"center",
                }),
                html.Div([
                    html.Span("Two-Wheeler", style={"color":"#dceeff"}),
                    html.Span(" Safety", style={"color":"#00e5ff"}),
                ], style={"fontFamily":"Orbitron,monospace","fontSize":32,"fontWeight":900,
                          "letterSpacing":"1.5px","textAlign":"center"}),
                html.Div("ADVANCED TWO-WHEELER SAFETY INTELLIGENCE", style={
                    "fontFamily":"Share Tech Mono,monospace","fontSize":12,
                    "color":"#6a8aaa","letterSpacing":"2.5px","textAlign":"center"}),
                html.Div(style={"width":80,"height":1,"background":"#152436","margin":"4px 0"}),
                html.Div(style={"display":"flex","gap":28,"marginBottom":4}, children=[
                    html.Div(style={"textAlign":"center"}, children=[
                        html.Div("◉", style={"color":"#00e5ff","fontSize":14}),
                        html.Div("RADAR", style={"fontFamily":"Share Tech Mono,monospace","fontSize":8,"color":"#3a5570","marginTop":2}),
                    ]),
                    html.Div(style={"textAlign":"center"}, children=[
                        html.Div("◈", style={"color":"#00e676","fontSize":14}),
                        html.Div("CAMERA", style={"fontFamily":"Share Tech Mono,monospace","fontSize":8,"color":"#3a5570","marginTop":2}),
                    ]),
                    html.Div(style={"textAlign":"center"}, children=[
                        html.Div("⚡", style={"color":"#ffab00","fontSize":14}),
                        html.Div("FUSION", style={"fontFamily":"Share Tech Mono,monospace","fontSize":8,"color":"#3a5570","marginTop":2}),
                    ]),
                ]),
                html.Button("▶  START MONITORING", id="audio-unlock-btn", n_clicks=0, style={
                    "fontFamily":"Orbitron,monospace","fontSize":13,"fontWeight":700,
                    "color":"#050c14","background":"#00e5ff","border":"none",
                    "borderRadius":4,"padding":"14px 40px","cursor":"pointer",
                    "letterSpacing":"1.5px","boxShadow":"0 0 32px rgba(0,229,255,0.50)",
                }),
                dcc.Checklist(
                    id="startup-alarm-check",
                    options=[{"label": " Enable audio alerts (beeps + voice)", "value": "alarm"}],
                    value=[],
                    inputStyle={"marginRight": "8px"},
                    labelStyle={
                        "fontFamily":"Share Tech Mono,monospace","fontSize":10,
                        "color":"#6a8aaa","cursor":"pointer",
                    },
                    style={"marginTop":"4px"},
                ),
                html.Div("Uncheck to run silently · frames advance in order after start", style={
                    "fontFamily":"Share Tech Mono,monospace","fontSize":9,"color":"#3a5570",
                }),
            ]),
        ], style={
            "position":"fixed","top":0,"left":0,"right":0,"bottom":0,
            "background":"rgba(5,12,20,0.97)",
            "display":"flex","alignItems":"center","justifyContent":"center",
            "zIndex":99999,"backdropFilter":"blur(6px)",
        }),

        # ══ TOP STATUS BAR ════════════════════════════════════════════════════
        html.Div(id="status-bar", className="status-bar",
                 style=init_sb_style, children=[

            # Brand — animated two-wheel mark (full wordmark lives on the splash)
            html.Div(style={
                "paddingRight":10,"borderRight":f"1px solid {T['border']}",
                "display":"flex","alignItems":"center","justifyContent":"center",
            }, children=[
                _logo_img(42),
            ]),

            # Overall risk (hidden min-ttc data holder nested here so it never
            # occupies its own status-bar grid track)
            html.Div(style={"textAlign":"center"}, children=[
                html.Div("OVERALL RISK", style={"fontSize":14,"color":T["text"],"letterSpacing":"1.2px","marginBottom":5,"fontWeight":800}),
                html.Div(id="overall-risk-badge", children=init_risk_badge),
                html.Div(id="min-ttc-value", children=init_ttc_el, style={"display":"none"}),
            ]),

            html.Div(className="top-gauge-strip", children=[
                _gauge_cell("MIN TTC",     "Time to collision", "gauge-ttc",      g1),
                _gauge_cell("HEADWAY",     "Time gap ahead",    "gauge-conflict", g2),
                _gauge_cell("REQ DECEL",   "Brake to avoid",    "gauge-uncert",   g3,
                            extra_class="gauge-cell--req-decel"),
                _gauge_cell("STOP MARGIN", "Clear road beyond braking", "gauge-conf", g4),
                _gauge_cell("CROWD",       "Objects ahead",     "gauge-crowd",    g5),
            ]),

            # Closest object
            html.Div(style={"textAlign":"center"}, children=[
                html.Div("CLOSEST OBJECT", style={"fontSize":12,"color":T["dim"],"letterSpacing":"1px","marginBottom":3,"fontWeight":700}),
                html.Div(id="closest-obj-text", children=init_cl_text),
                html.Div(id="closest-dist-value", children=init_cl_dist),
            ]),

            # High-risk count
            html.Div(style={"textAlign":"center"}, children=[
                html.Div("HIGH-RISK OBJECTS", style={"fontSize":12,"color":T["dim"],"letterSpacing":"1px","marginBottom":3,"fontWeight":700}),
                html.Div(id="high-risk-count", children=init_hr_el),
            ]),

            # Ego-motion average (configured; no odometry stream is loaded)
            html.Div(style={"textAlign":"center"}, children=[
                html.Div("EGO SPEED", style={"fontSize":12,"color":T["dim"],"letterSpacing":"1px","marginBottom":3,"fontWeight":700}),
                html.Div(id="ego-motion-value", children=init_ego_value,
                    style={"fontFamily":"Orbitron,monospace","fontSize":28,"fontWeight":800,"color":T["cyan"],"lineHeight":"1"}),
                html.Div(id="ego-motion-source", children=init_ego_source, style={"fontSize":11,"color":T["dim"]}),
            ]),

            # Frame number + sequence
            html.Div(style={"textAlign":"center"}, children=[
                html.Div("FRAME #", style={"fontSize":12,"color":T["dim"],"letterSpacing":"1px","marginBottom":3,"fontWeight":700}),
                html.Div(id="frame-num-value", children=init_frame_num),
                html.Div(id="frame-calib-value", children=init_frame_calib,
                    style={"fontSize":11,"color":T["dim"],"fontFamily":"Share Tech Mono,monospace","marginTop":3},
                ),
            ]),

            # Live clock + weather chip
            html.Div(style={"textAlign":"right","borderLeft":f"1px solid {T['border']}","paddingLeft":14,
                            "display":"flex","flexDirection":"column","alignItems":"flex-end","gap":5}, children=[
                html.Div(init_time, id="time-display",
                    style={"fontFamily":"Orbitron,monospace","fontSize":28,"color":T["bright"],
                           "fontWeight":800,"letterSpacing":"1px","lineHeight":"1"}),
                html.Div(init_date, id="date-display",
                    style={"fontSize":13,"color":T["dim"],"fontFamily":"Share Tech Mono,monospace"}),
                html.Div(init_weather_el, id="weather-status", style={"marginTop":3}),
            ]),
        ]),

        # ══ ROW 1: Camera  |  BEV  |  Occupancy + Legend ════════════════════
        html.Div(className="main-grid", children=[

            panel("CAMERA VIEW — FRONT", badge="● LIVE FUSED", badge_color=T["cyan"],
                  className="primary-visual-panel", children=[
                html.Div(style={"padding":"8px 12px","display":"flex","justifyContent":"flex-end","alignItems":"center"}, children=[
                    html.Div(className="video-controls", children=[
                    html.Div(className="speed-control", children=[
                        html.Span("SPEED", className="speed-label"),
                        dcc.RadioItems(
                            id="playback-speed",
                            options=[
                                {"label": "Slow", "value": "slow"},
                                {"label": "1x", "value": "1x"},
                                {"label": "2x", "value": "2x"},
                                {"label": "4x", "value": "4x"},
                            ],
                            value="1x",
                            className="speed-radio",
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"display":"inline-flex","alignItems":"center","gap":"2px"},
                        ),
                    ]),
                    html.Button("🔕 Alarm OFF", id="alarm-toggle-btn", n_clicks=0, style={
                        "fontFamily":"Share Tech Mono,monospace","fontSize":10,"color":"#ffffff",
                        "background":"#33506e","border":"none","borderRadius":5,"padding":"6px 12px",
                        "cursor":"pointer",
                        "transition":"background 120ms ease",
                    }),
                    html.Button("⏸ Pause", id="pause-button", n_clicks=0, style={
                        "fontFamily":"Share Tech Mono,monospace","fontSize":10,"color":"#ffffff",
                        "background":T["red"],"border":"none","borderRadius":5,"padding":"6px 12px",
                        "cursor":"pointer","boxShadow":"0 0 12px rgba(255,23,68,0.24)",
                        "transition":"background 120ms ease, transform 120ms ease",
                    }),
                    ]),
                ]),
                dcc.Graph(id="cam-graph", className="primary-graph", figure=cam_fig,
                          config=_GRAPH_CFG_VISUAL, responsive=True,
                          style={"height":"100%","minHeight":0,"flex":"1 1 auto"}),
            ]),

            panel("RADAR 3D ROAD VIEW",
                  className="primary-visual-panel", children=[
                dcc.Graph(id="bev-graph", className="primary-graph", figure=bev_fig,
                          config=_GRAPH_CFG_VISUAL, responsive=True,
                          style={"height":"100%","minHeight":0,"flex":"1 1 auto"}),
            ]),

            # Third column: SAFETY METRICS above EVENT LOG
            html.Div(className="visual-side-stack", children=[
                panel("SAFETY METRICS", className="side-stack-panel", children=[
                    html.Div(id="occupancy-bars",
                             style={"padding":"12px 14px","overflowY":"auto","height":"100%",
                                    "flex":"1 1 auto","minHeight":0},
                             children=[
                        occ_bar(label, val, lv)
                        for label, val, lv in compute_occupancy_metrics(init_objs)
                    ]),
                ]),
                panel("EVENT LOG", badge="● LIVE", badge_color=T["red"],
                      className="side-stack-panel", children=[
                    html.Div(id="event-log",
                        style={"padding":6,"overflowY":"auto","height":"100%",
                               "flex":"1 1 auto","minHeight":0},
                        children=init_events),
                ]),
            ]),

        ]),

        # ══ ROW 2: Object Table  |  Timeline  |  Risk Summary ═════════════════
        html.Div(className="row2-grid", children=[

            panel("OBJECT RISK TABLE", badge=f"{len(init_objs)} TRACKED",
                  badge_color=T["cyan"], badge_id="tracked-count-badge", children=[
                html.Div(id="object-table", style={"overflowX":"auto"}, children=init_table),
            ]),

            panel("TTC / RISK TIMELINE — LAST 30 s", children=[
                dcc.Graph(id="timeline-graph", className="primary-graph", figure=tl_fig,
                          config=_GRAPH_CFG, responsive=True,
                          style={"height":"100%","minHeight":0,"flex":"1 1 auto"}),
            ]),

            panel("RISK SUMMARY", className="risk-summary-panel", children=[
                html.Div(id="risk-summary", style={"padding":"10px 12px","overflowY":"auto"},
                         children=init_summary),
            ]),
        ]),
    ])

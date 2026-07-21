"""
app.py â€” Entry point. Run with:  python app.py
Then open:  http://localhost:8050
"""

import os
import time
from dash import Dash
from flask import abort, send_file
from data import AVAILABLE_FRAMES, get_frame_image_path
from playback_cache import warm_playback_figure_cache_async
from layout import create_layout
from callbacks import register_callbacks

app = Dash(
    __name__,
    title="twowheelersSafety",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
)


@app.server.after_request
def _no_cache_dash_internals(response):
    """Prevent browsers from caching Dash's dependency/layout descriptors.

    Without this, a tab opened before a callback-signature change keeps posting
    the stale signature, which the server rejects â€” flooding logs and stalling
    playback until a hard refresh. Forcing revalidation keeps clients in sync.
    """
    from flask import request
    if request.path.startswith(("/_dash-dependencies", "/_dash-layout")):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


_STALE_WARN_STATE = {"last": 0.0}


@app.server.before_request
def _reject_stale_callbacks():
    """Quietly reject callback requests from stale browser tabs.

    A tab loaded before a callback-signature change keeps posting the old
    signature. Left to Dash, each one raises a KeyError with a ~40-line
    traceback; a tab polling every few hundred ms saturates the single-threaded
    dev server and stalls playback for every client. Here we check the requested
    output against the registered callback map and, if it's unknown, return a
    tiny 410 (Gone) immediately â€” no exception, no traceback, negligible CPU â€”
    with a throttled one-line hint telling the user to refresh.
    """
    from flask import request, Response
    if request.path != "/_dash-update-component" or request.method != "POST":
        return None
    body = request.get_json(silent=True) or {}
    output = body.get("output")
    if output and output not in app.callback_map:
        now = time.time()
        if now - _STALE_WARN_STATE["last"] > 5.0:
            _STALE_WARN_STATE["last"] = now
            print("  [stale-tab] Ignoring outdated callback request â€” "
                  "hard-refresh that browser tab (Ctrl+Shift+R).", flush=True)
        return Response("stale callback signature â€” refresh the page", status=410)
    return None


@app.server.route("/frame-image/<frame_id>")
def frame_image(frame_id):
    if frame_id not in AVAILABLE_FRAMES:
        abort(404)
    image_path = get_frame_image_path(frame_id)
    if not os.path.exists(image_path):
        abort(404)
    return send_file(image_path, conditional=True, max_age=3600)


app.layout = create_layout()

# Optional video-export routes (only if export_snapshot.py is present).
try:
    from export_snapshot import build_export_snapshot_html, build_splash_snapshot_html
    from flask import Response

    @app.server.route("/export/frame/<int:frame_index>")
    def export_frame(frame_index):
        html_content = build_export_snapshot_html(frame_index, css_href="/assets/custom.css")
        return Response(html_content, mimetype="text/html")

    @app.server.route("/export/splash")
    def export_splash():
        html_content = build_splash_snapshot_html(css_href="/assets/custom.css")
        return Response(html_content, mimetype="text/html")
except ImportError:
    pass

register_callbacks(app)

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "8050"))
    print("\n  Two-Wheeler Safety Dashboard")
    print(f"  Data:  {__import__('data').KITTI_BASE_PATH}")
    print(f"  Open:  http://localhost:{port}\n")
    # Pre-build a lookahead window so the first playback ticks dequeue ready bundles.
    warm_playback_figure_cache_async(0)
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False, threaded=True)

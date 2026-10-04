import requests
import numpy as np
import os
import time
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output, State, Patch, ctx
from uncertainty_model import combined_sigma, tle_age_hours, probability_of_collision

MU_EARTH_KM3S2 = 398600.4418
R_EARTH = 6371.0
BIN_WIDTH_KM = 25
MAX_PER_BUCKET = 400
COARSE_CANDIDATE_KM = 200
MAX_CANDIDATES = 300
WINDOW_MINUTES = 180
COARSE_STEP_MIN = 12
FINE_STEP_MIN = 1
HARD_BODY_RADIUS_M = 10.0
MAX_RISK_SLOTS = 10
LOCK_TRACE_IDX = 4 + MAX_RISK_SLOTS       # 14
ORBIT_TRACE_IDX = LOCK_TRACE_IDX + 1      # 15
STARFIELD_EXTENT = 48000.0                # matches the starfield radius used below

CACHE_FILE = "tle_cache.txt"
CACHE_MAX_AGE_HOURS = 2

print("Building textured Earth...")
EARTH_TEXTURE_URL = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg"
img = Image.open(BytesIO(requests.get(EARTH_TEXTURE_URL).content))
GRID_W, GRID_H = 140, 70
N_COLORS = 180
img_small = img.resize((GRID_W, GRID_H), Image.LANCZOS).convert("RGB")
img_q = img_small.quantize(colors=N_COLORS, method=Image.MEDIANCUT)
_palette = img_q.getpalette()[: N_COLORS * 3]
_palette_rgb = [tuple(_palette[i : i + 3]) for i in range(0, len(_palette), 3)]
EARTH_INDEX_GRID = np.array(img_q).astype(float)
EARTH_COLORSCALE = [[i / (N_COLORS - 1), f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(_palette_rgb)]
_lon = np.linspace(0, 2 * np.pi, GRID_W)
_colat = np.linspace(0, np.pi, GRID_H)
_LON, _COLAT = np.meshgrid(_lon, _colat)
EARTH_X = R_EARTH * np.cos(_LON) * np.sin(_COLAT)
EARTH_Y = R_EARTH * np.sin(_LON) * np.sin(_COLAT)
EARTH_Z = R_EARTH * np.cos(_COLAT)
_gu, _gv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
GLOW_R = R_EARTH * 1.035
GLOW_X = GLOW_R * np.cos(_gu) * np.sin(_gv)
GLOW_Y = GLOW_R * np.sin(_gu) * np.sin(_gv)
GLOW_Z = GLOW_R * np.cos(_gv)
print("Earth ready.")


def fetch_tle_text():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    if os.path.exists(CACHE_FILE):
        age_h = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600
        if age_h < CACHE_MAX_AGE_HOURS:
            print(f"Using cached TLE data ({age_h:.2f}h old)...")
            with open(CACHE_FILE) as f:
                return f.read()
    print("Fetching fresh active-satellite catalog from CelesTrak...")
    resp = requests.get(url, timeout=20)
    text = resp.text
    lines = text.strip().splitlines()
    if resp.status_code != 200 or len(lines) < 3:
        print(f"WARNING: fetch looks wrong (HTTP {resp.status_code}, {len(lines)} lines).")
        if os.path.exists(CACHE_FILE):
            print("Falling back to the existing cache.")
            with open(CACHE_FILE) as f:
                return f.read()
        raise RuntimeError("TLE fetch failed and no cache exists yet. Wait ~10-15 minutes and try again.")
    with open(CACHE_FILE, "w") as f:
        f.write(text)
    return text


print("Loading satellite catalog...")
raw_text = fetch_tle_text()
lines = raw_text.strip().splitlines()
NAMES, SATS = [], []
for i in range(0, len(lines) - 2, 3):
    NAMES.append(lines[i].strip())
    SATS.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
print(f"Loaded {len(SATS)} satellites.")
if len(SATS) == 0:
    raise RuntimeError("Parsed 0 satellites — check raw_text[:500] to inspect the response.")

NAMES_LOWER = [n.lower() for n in NAMES]


def cw_delta_r(n, t, dv_rsw):
    nt = n * t
    Phi = np.array([
        [(1 / n) * np.sin(nt), (2 / n) * (1 - np.cos(nt)), 0],
        [(2 / n) * (np.cos(nt) - 1), (1 / n) * (4 * np.sin(nt) - 3 * n * t), 0],
        [0, 0, (1 / n) * np.sin(nt)],
    ])
    return Phi @ dv_rsw


def action_recommendation(pc):
    if pc > 1e-4:
        return "CRITICAL: Execute Avoidance Maneuver", "#ff3b3b"
    if pc > 1e-6:
        return "WARNING: Monitor TLE", "#ffa63b"
    return "NOMINAL: Orbit Clear", "#5cff8f"


def plain_language_risk(pc):
    if pc > 1e-4:
        return "HIGH", "#ff3b3b", "These two objects are close enough, and moving fast enough, that a collision is a real possibility."
    if pc > 1e-6:
        return "MODERATE", "#ffa63b", "These two objects will pass close to each other — worth continuing to track."
    return "LOW", "#5cff8f", "These two objects will miss each other by a comfortable margin."


def find_satellite(query):
    """Case-insensitive substring search. Returns (index, name) of the first match, or (None, None)."""
    q = query.strip().lower()
    if not q:
        return None, None
    if q.isdigit():
        for i, sat in enumerate(SATS):
            if str(sat.satnum) == q:
                return i, NAMES[i]
    for i, n in enumerate(NAMES_LOWER):
        if q in n:
            return i, NAMES[i]
    return None, None


def satellite_telemetry(idx, jd, fr):
    sat = SATS[idx]
    err, pos, vel = sat.sgp4(jd, fr)
    pos, vel = np.array(pos), np.array(vel)
    altitude_km = np.linalg.norm(pos) - R_EARTH
    speed_kms = np.linalg.norm(vel)
    inclination_deg = np.degrees(sat.inclo)
    period_min = 2 * np.pi / sat.no_kozai
    return dict(name=NAMES[idx], norad_id=sat.satnum, altitude_km=altitude_km,
                speed_kms=speed_kms, inclination_deg=inclination_deg,
                period_min=period_min, pos=pos)


def satellite_orbit_path(idx, now, minutes=100, step=1):
    sat = SATS[idx]
    pts = []
    for m in range(0, minutes, step):
        t = now + timedelta(minutes=m)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        err, pos, vel = sat.sgp4(jd, fr)
        if err == 0:
            pts.append(pos)
    return np.array(pts)


def build_static_traces():
    rs = np.random.default_rng(42)
    n_stars = 900
    sr = rs.uniform(32000, STARFIELD_EXTENT, n_stars)
    st = rs.uniform(0, 2 * np.pi, n_stars)
    sp = rs.uniform(0, np.pi, n_stars)
    starfield = go.Scatter3d(
        x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
        mode="markers", marker=dict(size=1.2, color="white", opacity=0.55),
        hoverinfo="skip", showlegend=False)
    glow = go.Surface(x=GLOW_X, y=GLOW_Y, z=GLOW_Z,
        colorscale=[[0, "#3fa9ff"], [1, "#3fa9ff"]], showscale=False, opacity=0.12, hoverinfo="skip")
    earth = go.Surface(x=EARTH_X, y=EARTH_Y, z=EARTH_Z,
        surfacecolor=EARTH_INDEX_GRID, colorscale=EARTH_COLORSCALE,
        cmin=0, cmax=N_COLORS - 1, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.9))
    return [starfield, glow, earth]


def build_swarm_trace(current_pos, valid):
    return go.Scatter3d(
        x=current_pos[valid, 0], y=current_pos[valid, 1], z=current_pos[valid, 2],
        mode="markers", marker=dict(size=1.6, color="#7fd7ff", opacity=0.45),
        hoverinfo="skip", name=f"{valid.sum()} tracked satellites", showlegend=False)


def compute_dynamic():
    now = datetime.now(timezone.utc)
    jd0, fr0 = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)

    current_pos = np.zeros((len(SATS), 3))
    valid = np.ones(len(SATS), dtype=bool)
    for i, sat in enumerate(SATS):
        err, pos, vel = sat.sgp4(jd0, fr0)
        if err == 0:
            current_pos[i] = pos
        else:
            valid[i] = False

    altitude = np.linalg.norm(current_pos, axis=1) - R_EARTH
    bins = np.floor(altitude / BIN_WIDTH_KM).astype(int)
    bins[~valid] = -999999

    coarse_times = [now + timedelta(minutes=m) for m in range(0, WINDOW_MINUTES, COARSE_STEP_MIN)]
    coarse_jdfr = [jday(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in coarse_times]

    rng = np.random.default_rng(0)
    candidates = []
    for b in np.unique(bins[valid]):
        idxs = np.where(bins == b)[0]
        if len(idxs) < 2:
            continue
        if len(idxs) > MAX_PER_BUCKET:
            idxs = rng.choice(idxs, size=MAX_PER_BUCKET, replace=False)
        pos_c = np.zeros((len(idxs), len(coarse_times), 3))
        for bi, si in enumerate(idxs):
            for ti, (jd, fr) in enumerate(coarse_jdfr):
                err, pos, vel = SATS[si].sgp4(jd, fr)
                pos_c[bi, ti] = pos
        min_dist = np.full((len(idxs), len(idxs)), np.inf)
        for t in range(len(coarse_times)):
            pt = pos_c[:, t, :]
            d = np.linalg.norm(pt[:, None, :] - pt[None, :, :], axis=-1)
            min_dist = np.minimum(min_dist, d)
        np.fill_diagonal(min_dist, np.inf)
        iu, ju = np.triu_indices(len(idxs), k=1)
        mask = min_dist[iu, ju] < COARSE_CANDIDATE_KM
        for a, c, dist in zip(iu[mask], ju[mask], min_dist[iu, ju][mask]):
            candidates.append((idxs[a], idxs[c], dist))

    candidates.sort(key=lambda c: c[2])
    candidates = candidates[:MAX_CANDIDATES]

    fine_times = [now + timedelta(minutes=m) for m in range(0, WINDOW_MINUTES, FINE_STEP_MIN)]
    fine_jdfr = [jday(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in fine_times]

    refined = []
    for i, j, _ in candidates:
        pa, pb, va, vb = [], [], [], []
        for jd, fr in fine_jdfr:
            e1, p1, v1 = SATS[i].sgp4(jd, fr)
            e2, p2, v2 = SATS[j].sgp4(jd, fr)
            pa.append(p1); pb.append(p2); va.append(v1); vb.append(v2)
        pa, pb, va, vb = map(np.array, (pa, pb, va, vb))
        d = np.linalg.norm(pa - pb, axis=1)
        idx = int(np.argmin(d))
        refined.append(dict(i=i, j=j, a=NAMES[i], b=NAMES[j], miss_km=d[idx],
                             rel_v=float(np.linalg.norm(va[idx] - vb[idx])),
                             pa=pa[idx], pb=pb[idx], tca_idx=idx))

    if len(refined) >= 5:
        X = np.array([[r["miss_km"], r["rel_v"]] for r in refined])
        Xs = StandardScaler().fit_transform(X)
        model = IsolationForest(contamination=min(0.2, 8 / len(refined)), random_state=42).fit(Xs)
        anomaly = -model.score_samples(Xs)
        risk = 100 * (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)
        for r, s in zip(refined, risk):
            r["risk"] = s
        refined.sort(key=lambda r: -r["risk"])
    top_risks = refined[:MAX_RISK_SLOTS]

    for r in top_risks:
        age_a = tle_age_hours(SATS[r["i"]], jd0, fr0)
        age_b = tle_age_hours(SATS[r["j"]], jd0, fr0)
        sigma = combined_sigma(age_a, age_b)
        r["pc"] = probability_of_collision(r["miss_km"] * 1000, sigma, HARD_BODY_RADIUS_M)
        r["sigma_used"] = sigma

    top_pair_data = None
    if top_risks:
        top = top_risks[0]
        tca_seconds_from_now = top["tca_idx"] * FINE_STEP_MIN * 60
        lead_s = min(90 * 60, tca_seconds_from_now)
        burn_time = now + timedelta(seconds=(tca_seconds_from_now - lead_s))
        jd_b, fr_b = jday(burn_time.year, burn_time.month, burn_time.day,
                           burn_time.hour, burn_time.minute, burn_time.second + burn_time.microsecond / 1e6)
        _, r0_burn, v0_burn = SATS[top["i"]].sgp4(jd_b, fr_b)
        top_pair_data = dict(
            a=top["a"], b=top["b"], risk=top["risk"], pc=top["pc"],
            r0=list(r0_burn), v0=list(v0_burn), r_rel_km=(top["pa"] - top["pb"]).tolist(),
            t_flight_s=lead_s, lead_available_s=tca_seconds_from_now,
        )

    risk_slots = []
    for r in top_risks:
        label = (f"{r['a']} vs {r['b']}<br>Miss: {r['miss_km']:.2f} km<br>"
                 f"Anomaly rank: {r['risk']:.0f}/100<br>Pc: {r['pc']:.2e} (sigma={r['sigma_used']:.0f} m)")
        color = "#ff3b3b" if r["risk"] > 60 else "#ffa63b"
        risk_slots.append(dict(x=[r["pa"][0], r["pb"][0]], y=[r["pa"][1], r["pb"][1]], z=[r["pa"][2], r["pb"][2]],
                                color=color, text=[label, label], name=f"Risk {r['risk']:.0f}"))
    while len(risk_slots) < MAX_RISK_SLOTS:
        risk_slots.append(dict(x=[], y=[], z=[], color="#ff2b2b", text=[], name=""))

    stats = dict(
        count=int(valid.sum()),
        flagged=len([r for r in top_risks if r["risk"] > 60]),
        closest=min((r["miss_km"] for r in top_risks), default=float("nan")),
        max_pc=max((r["pc"] for r in top_risks), default=0.0),
        updated=now.strftime("%H:%M:%S UTC"),
        top_pair=top_pair_data,
    )
    return current_pos, valid, risk_slots, stats, now, jd0, fr0


HUD_STYLE = {
    "position": "absolute", "top": "18px", "left": "18px", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.75)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "14px 20px", "color": "#7fd7ff",
    "fontFamily": "Consolas, monospace", "fontSize": "14px", "lineHeight": "1.8",
    "boxShadow": "0 0 18px rgba(63,169,255,0.25)",
}
SANDBOX_STYLE = {
    "position": "absolute", "top": "18px", "right": "18px", "width": "340px", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.85)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "16px 20px", "color": "#7fd7ff",
    "fontFamily": "Consolas, monospace", "fontSize": "12px",
    "boxShadow": "0 0 18px rgba(63,169,255,0.25)",
}
CONTROL_BAR_STYLE = {
    "position": "absolute", "bottom": "18px", "left": "18px", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.8)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "10px 16px", "color": "#7fd7ff",
    "fontFamily": "Consolas, monospace", "fontSize": "12px", "display": "flex",
    "gap": "18px", "alignItems": "center", "flexWrap": "wrap",
}
LEGEND_STYLE = {
    "position": "absolute", "bottom": "18px", "right": "18px", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.8)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "10px 16px", "color": "#c9e8ff",
    "fontFamily": "Consolas, monospace", "fontSize": "11px", "lineHeight": "1.9",
}
SEARCH_STYLE = {
    "position": "absolute", "top": "18px", "left": "50%", "transform": "translateX(-50%)", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.85)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "10px 16px", "display": "flex", "gap": "8px", "alignItems": "center",
}
TELEMETRY_STYLE_BASE = {
    "position": "absolute", "top": "90px", "left": "50%", "transform": "translateX(-50%)", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.85)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "12px 20px", "color": "#7fd7ff",
    "fontFamily": "Consolas, monospace", "fontSize": "12px", "lineHeight": "1.7",
}


def dot(color):
    return html.Span("●", style={"color": color, "marginRight": "6px"})


def legend_children():
    return [
        html.Div("LEGEND", style={"fontWeight": "bold", "marginBottom": "6px", "color": "#fff"}),
        html.Div([dot("#7fd7ff"), "Tracked satellite"]),
        html.Div([dot("#ffa63b"), "Moderate-risk conjunction"]),
        html.Div([dot("#ff3b3b"), "High-risk conjunction"]),
        html.Div([dot("#fff44f"), "Searched / locked satellite"]),
    ]


_static = build_static_traces()
_init_pos, _init_valid, _init_risk_slots, _init_stats, _init_now, _, _ = compute_dynamic()
_lock_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers",
                            marker=dict(size=7, color="#fff44f", symbol="diamond"),
                            hoverinfo="text", text=[], name="Locked satellite", showlegend=False)
_orbit_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines",
                             line=dict(color="#fff44f", width=3, dash="dot"),
                             hoverinfo="skip", name="Locked satellite orbit", showlegend=False)
_initial_fig = go.Figure(data=_static + [build_swarm_trace(_init_pos, _init_valid)] + [
    go.Scatter3d(x=s["x"], y=s["y"], z=s["z"], mode="lines+markers",
                 line=dict(color=s["color"], width=7), marker=dict(size=5, color=s["color"]),
                 text=s["text"], hoverinfo="text", name=s["name"], showlegend=False)
    for s in _init_risk_slots
] + [_lock_trace, _orbit_trace])
_initial_fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               bgcolor="black", aspectmode="data"),
    paper_bgcolor="black", font=dict(color="#c9e8ff", family="Consolas, monospace"),
    showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
    uirevision="keep",
)

app = Dash(__name__)
app.layout = html.Div(style={"backgroundColor": "black", "height": "100vh", "position": "relative"}, children=[
    html.Div(id="hud", style=HUD_STYLE),
    html.Div(id="legend", style=LEGEND_STYLE, children=legend_children()),

    html.Div(style=SEARCH_STYLE, children=[
        dcc.Input(id="search-input", type="text", placeholder="Search satellite name or NORAD ID...",
                  style={"width": "260px", "padding": "6px", "borderRadius": "4px", "border": "1px solid #2b6a99"},
                  debounce=False, n_submit=0),
        html.Button("Lock", id="search-btn", n_clicks=0,
                    style={"backgroundColor": "#16233d", "color": "#7fd7ff", "border": "1px solid #2b6a99",
                           "borderRadius": "4px", "padding": "6px 12px", "cursor": "pointer"}),
        dcc.Checklist(id="orbit-toggle", options=[{"label": " Show full orbit", "value": "on"}], value=[],
                      style={"color": "#7fd7ff", "fontSize": "12px", "marginLeft": "8px"}),
        html.Div(id="search-feedback", style={"color": "#ffa63b", "fontSize": "11px", "marginLeft": "8px"}),
    ]),
    html.Div(id="telemetry-card", style={**TELEMETRY_STYLE_BASE, "display": "none"}),

    html.Div(id="sandbox", style=SANDBOX_STYLE, children=[
        html.Div([
            html.Div("MODE", style={"fontSize": "10px", "opacity": 0.6, "marginBottom": "4px"}),
            dcc.RadioItems(
                id="view-mode",
                options=[{"label": " Student", "value": "student"}, {"label": " Advanced", "value": "advanced"}],
                value="student", labelStyle={"display": "inline-block", "marginRight": "14px"},
            ),
        ], style={"marginBottom": "14px", "borderBottom": "1px solid #2b6a99", "paddingBottom": "10px"}),
        html.Div(id="sandbox-target", style={"marginBottom": "10px", "opacity": 0.85}),
        html.Div(id="student-panel", children=[html.Div(id="student-output")]),
        html.Div(id="advanced-panel", style={"display": "none"}, children=[
            dcc.RadioItems(
                id="mode-toggle",
                options=[{"label": " Physics Mode", "value": "physics"}, {"label": " AI Mode", "value": "ai"}],
                value="physics", labelStyle={"display": "block", "marginBottom": "4px"},
                style={"marginBottom": "12px"},
            ),
            html.Div("Thruster burn (Δv, m/s):", style={"marginBottom": "4px"}),
            dcc.Slider(id="dv-slider", min=-5, max=5, step=0.1, value=0,
                       marks={-5: "-5", 0: "0", 5: "+5"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div("Tracking uncertainty (σ, m):", style={"marginTop": "14px", "marginBottom": "4px"}),
            dcc.Slider(id="sigma-slider", min=100, max=2000, step=50, value=1000,
                       marks={100: "100", 1000: "1000", 2000: "2000"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(id="sandbox-output", style={"marginTop": "16px", "lineHeight": "1.8"}),
        ]),
    ]),

    html.Div(id="control-bar", style=CONTROL_BAR_STYLE, children=[
        html.Button("⏸ Pause Live Refresh", id="pause-btn", n_clicks=0,
                    style={"backgroundColor": "#16233d", "color": "#7fd7ff", "border": "1px solid #2b6a99",
                           "borderRadius": "4px", "padding": "6px 12px", "cursor": "pointer",
                           "fontFamily": "Consolas, monospace", "fontSize": "12px"}),
        html.Div("Refresh:"),
        dcc.Dropdown(id="interval-choice",
                     options=[{"label": "15s", "value": 15}, {"label": "30s", "value": 30}, {"label": "60s", "value": 60}],
                     value=30, clearable=False, style={"width": "80px", "color": "#000"}),
    ]),

    dcc.Graph(id="live-graph", figure=_initial_fig, style={"height": "100vh"}, config={"displayModeBar": False}),
    dcc.Interval(id="refresh", interval=30 * 1000, n_intervals=0),
    dcc.Store(id="top-pair-store", data=_init_stats["top_pair"]),
    dcc.Store(id="locked-sat-store", data=None),
])


@app.callback(Output("refresh", "disabled"), Input("pause-btn", "n_clicks"))
def toggle_pause(n_clicks):
    return n_clicks % 2 == 1


@app.callback(Output("pause-btn", "children"), Input("pause-btn", "n_clicks"))
def pause_label(n_clicks):
    return "▶ Resume Live Refresh" if n_clicks % 2 == 1 else "⏸ Pause Live Refresh"


@app.callback(Output("refresh", "interval"), Input("interval-choice", "value"))
def set_interval(seconds):
    return seconds * 1000


@app.callback(Output("student-panel", "style"), Output("advanced-panel", "style"), Input("view-mode", "value"))
def toggle_view_mode(mode):
    if mode == "student":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@app.callback(
    Output("locked-sat-store", "data"), Output("search-feedback", "children"),
    Input("search-btn", "n_clicks"), Input("search-input", "n_submit"),
    State("search-input", "value"),
)
def do_search(n_clicks, n_submit, query):
    if not query:
        return None, ""
    idx, name = find_satellite(query)
    if idx is None:
        return None, f'No match for "{query}"'
    return idx, f"Found: {name}"


@app.callback(
    Output("live-graph", "figure"), Output("hud", "children"), Output("top-pair-store", "data"),
    Output("telemetry-card", "children"), Output("telemetry-card", "style"),
    Input("refresh", "n_intervals"), Input("locked-sat-store", "data"), Input("orbit-toggle", "value"),
)
def update(n, locked_idx, orbit_toggle):
    current_pos, valid, risk_slots, stats, now, jd0, fr0 = compute_dynamic()
    hud = [
        html.Div("ORBITSENSE — LIVE", style={"fontSize": "16px", "fontWeight": "bold", "color": "#fff", "marginBottom": "6px"}),
        html.Div(f"TRACKED OBJECTS ..... {stats['count']:,}"),
        html.Div(f"FLAGGED CONJUNCTIONS  {stats['flagged']}", style={"color": "#ff5c5c" if stats["flagged"] else "#7fd7ff"}),
        html.Div(f"CLOSEST APPROACH .... {stats['closest']:.2f} km" if stats['closest'] == stats['closest'] else "CLOSEST APPROACH .... --"),
        html.Div(f"HIGHEST Pc .......... {stats['max_pc']:.2e}", style={"color": "#ffa63b"}),
        html.Div(f"LAST UPDATE ......... {stats['updated']}", style={"opacity": 0.7, "marginTop": "6px"}),
    ]

    telem_children, telem_style = [], {**TELEMETRY_STYLE_BASE, "display": "none"}
    lock_x, lock_y, lock_z, lock_text = [], [], [], []
    orbit_x, orbit_y, orbit_z = [], [], []

    if locked_idx is not None:
        t = satellite_telemetry(locked_idx, jd0, fr0)
        lock_x, lock_y, lock_z = [t["pos"][0]], [t["pos"][1]], [t["pos"][2]]
        lock_text = [f"{t['name']} (NORAD {t['norad_id']})"]
        telem_style = {**TELEMETRY_STYLE_BASE, "display": "block"}
        telem_children = [
            html.Div(f"{t['name']}  (NORAD {t['norad_id']})", style={"fontWeight": "bold", "color": "#fff44f", "marginBottom": "4px"}),
            html.Span(f"ALT {t['altitude_km']:.0f} km   ", ),
            html.Span(f"SPD {t['speed_kms']:.2f} km/s   "),
            html.Span(f"INC {t['inclination_deg']:.1f}°   "),
            html.Span(f"PERIOD {t['period_min']:.1f} min"),
        ]
        if orbit_toggle and "on" in orbit_toggle:
            path = satellite_orbit_path(locked_idx, now)
            if len(path):
                orbit_x, orbit_y, orbit_z = path[:, 0].tolist(), path[:, 1].tolist(), path[:, 2].tolist()

    is_first_load = (ctx.triggered_id is None) or (n == 0 and ctx.triggered_id == "refresh")
    is_new_lock = ctx.triggered_id == "locked-sat-store"

    if is_first_load and not is_new_lock:
        return _initial_fig, hud, stats["top_pair"], telem_children, telem_style

    patched = Patch()
    patched["data"][3]["x"] = current_pos[valid, 0]
    patched["data"][3]["y"] = current_pos[valid, 1]
    patched["data"][3]["z"] = current_pos[valid, 2]
    patched["data"][3]["name"] = f"{valid.sum()} tracked satellites"
    for idx, s in enumerate(risk_slots):
        ti = 4 + idx
        patched["data"][ti]["x"] = s["x"]
        patched["data"][ti]["y"] = s["y"]
        patched["data"][ti]["z"] = s["z"]
        patched["data"][ti]["line"]["color"] = s["color"]
        patched["data"][ti]["marker"]["color"] = s["color"]
        patched["data"][ti]["text"] = s["text"]
        patched["data"][ti]["name"] = s["name"]

    patched["data"][LOCK_TRACE_IDX]["x"] = lock_x
    patched["data"][LOCK_TRACE_IDX]["y"] = lock_y
    patched["data"][LOCK_TRACE_IDX]["z"] = lock_z
    patched["data"][LOCK_TRACE_IDX]["text"] = lock_text
    patched["data"][ORBIT_TRACE_IDX]["x"] = orbit_x
    patched["data"][ORBIT_TRACE_IDX]["y"] = orbit_y
    patched["data"][ORBIT_TRACE_IDX]["z"] = orbit_z

    # Only move the camera when a NEW satellite is locked — never on a routine refresh,
    # so the user's manual zoom/pan is preserved while a lock is active.
    if is_new_lock and locked_idx is not None:
        pos = np.array([lock_x[0], lock_y[0], lock_z[0]])
        direction = pos / np.linalg.norm(pos)
        # Best-effort camera framing — Plotly's 3D camera coordinates are normalized to the
        # scene's bounding box, not raw data units, so this may need one round of tuning
        # based on what it actually looks like once rendered.
        eye = direction * 0.35
        center = direction * 0.28
        patched["layout"]["scene"]["camera"] = dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                                                      center=dict(x=center[0], y=center[1], z=center[2]))

    return patched, hud, stats["top_pair"], telem_children, telem_style


@app.callback(Output("sandbox-target", "children"), Input("top-pair-store", "data"))
def update_target_label(top):
    if not top:
        return "No flagged pair yet — waiting for first refresh."
    return f"Target: {top['a']}  vs  {top['b']}"


@app.callback(Output("student-output", "children"), Input("top-pair-store", "data"))
def update_student_panel(top):
    if not top:
        return ""
    level, color, explanation = plain_language_risk(top["pc"])
    return html.Div([
        html.Div(f"Risk level: {level}", style={"color": color, "fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"}),
        html.Div(explanation, style={"fontSize": "12px", "lineHeight": "1.6", "opacity": 0.9}),
        html.Div("Switch to Advanced mode to run your own what-if maneuvers.",
                 style={"fontSize": "10.5px", "opacity": 0.55, "marginTop": "12px", "fontStyle": "italic"}),
    ])


@app.callback(
    Output("sandbox-output", "children"),
    Input("dv-slider", "value"), Input("sigma-slider", "value"),
    Input("mode-toggle", "value"), Input("top-pair-store", "data"),
)
def update_sandbox(dv_m_s, sigma_m, mode, top):
    if not top:
        return ""
    lead_min = top["t_flight_s"] / 60
    low_lead_note = (
        html.Div(f"⚠ Only {lead_min:.0f} min of lead time available before this encounter.",
                 style={"color": "#ffa63b", "fontSize": "10.5px", "marginBottom": "8px"})
        if lead_min < 15 else None
    )
    if mode == "ai":
        return html.Div([
            html.Div(f"Isolation Forest anomaly rank: {top['risk']:.0f} / 100",
                      style={"color": "#7fd7ff", "fontSize": "14px"}),
            html.Div("(population-relative — doesn't move with these sliders.)",
                     style={"opacity": 0.6, "fontSize": "10.5px", "marginTop": "6px"}),
        ])
    r0 = np.array(top["r0"]); v0 = np.array(top["v0"])
    r_rel_baseline = np.array(top["r_rel_km"])
    R_hat = r0 / np.linalg.norm(r0)
    W_hat = np.cross(r0, v0); W_hat /= np.linalg.norm(W_hat)
    S_hat = np.cross(W_hat, R_hat)
    n = np.sqrt(MU_EARTH_KM3S2 / np.linalg.norm(r0) ** 3)
    dv_rsw = np.array([0, dv_m_s / 1000.0, 0])
    delta_r_rsw = cw_delta_r(n, top["t_flight_s"], dv_rsw)
    delta_r_eci = delta_r_rsw[0] * R_hat + delta_r_rsw[1] * S_hat + delta_r_rsw[2] * W_hat
    new_miss_km = np.linalg.norm(r_rel_baseline + delta_r_eci)
    pc = probability_of_collision(new_miss_km * 1000, sigma_m, HARD_BODY_RADIUS_M)
    rec_text, rec_color = action_recommendation(pc)
    return html.Div([
        low_lead_note,
        html.Div(f"Miss distance: {new_miss_km * 1000:.1f} m", style={"color": "#7fd7ff", "fontSize": "14px"}),
        html.Div(f"Pc: {pc:.3e}", style={"color": "#7fd7ff", "fontSize": "14px", "marginTop": "2px"}),
        html.Div(rec_text, style={"color": rec_color, "fontWeight": "bold", "marginTop": "10px", "fontSize": "13px"}),
    ])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
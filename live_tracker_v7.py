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
from dash import Dash, dcc, html, Input, Output, State, Patch, ctx, dash_table, no_update
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
LOCK_TRACE_IDX = 4 + MAX_RISK_SLOTS
ORBIT_TRACE_IDX = LOCK_TRACE_IDX + 1
STARFIELD_EXTENT = 40000.0
TIME_SCRUB_LIMIT_MIN = 180
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
            with open(CACHE_FILE) as f:
                return f.read()
        raise RuntimeError("TLE fetch failed and no cache exists yet.")
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
    raise RuntimeError("Parsed 0 satellites — check raw_text[:500].")
NAMES_LOWER = [n.lower() for n in NAMES]


def cw_delta_r(n, t, dv_rsw):
    nt = n * t
    Phi = np.array([
        [(1 / n) * np.sin(nt), (2 / n) * (1 - np.cos(nt)), 0],
        [(2 / n) * (np.cos(nt) - 1), (1 / n) * (4 * np.sin(nt) - 3 * n * t), 0],
        [0, 0, (1 / n) * np.sin(nt)],
    ])
    return Phi @ dv_rsw


def severity_tag(pc):
    if pc > 1e-4:
        return "CRITICAL", "#ef4444"
    if pc > 1e-6:
        return "MODERATE", "#f59e0b"
    return "NOMINAL", "#10b981"


def action_recommendation(pc):
    if pc > 1e-4:
        return "CRITICAL: Execute Avoidance Maneuver", "#ef4444"
    if pc > 1e-6:
        return "WARNING: Monitor TLE", "#f59e0b"
    return "NOMINAL: Orbit Clear", "#10b981"


def plain_language_risk(pc):
    if pc > 1e-4:
        return "HIGH", "#ef4444", "These two objects are close enough, and moving fast enough, that a collision is a real possibility."
    if pc > 1e-6:
        return "MODERATE", "#f59e0b", "These two objects will pass close to each other — worth continuing to track."
    return "LOW", "#10b981", "These two objects will miss each other by a comfortable margin."


def find_satellite(query):
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
    return dict(name=NAMES[idx], norad_id=sat.satnum, altitude_km=np.linalg.norm(pos) - R_EARTH,
                speed_kms=np.linalg.norm(vel), inclination_deg=np.degrees(sat.inclo),
                period_min=2 * np.pi / sat.no_kozai, pos=pos)


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
    n_stars = 800
    sr = rs.uniform(26000, STARFIELD_EXTENT, n_stars)
    st = rs.uniform(0, 2 * np.pi, n_stars)
    sp = rs.uniform(0, np.pi, n_stars)
    starfield = go.Scatter3d(x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
        mode="markers", marker=dict(size=1.1, color="white", opacity=0.5), hoverinfo="skip", showlegend=False)
    glow = go.Surface(x=GLOW_X, y=GLOW_Y, z=GLOW_Z, colorscale=[[0, "#00f0ff"], [1, "#00f0ff"]],
        showscale=False, opacity=0.1, hoverinfo="skip")
    earth = go.Surface(x=EARTH_X, y=EARTH_Y, z=EARTH_Z, surfacecolor=EARTH_INDEX_GRID, colorscale=EARTH_COLORSCALE,
        cmin=0, cmax=N_COLORS - 1, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.9))
    return [starfield, glow, earth]


def build_swarm_trace(current_pos, valid):
    return go.Scatter3d(x=current_pos[valid, 0], y=current_pos[valid, 1], z=current_pos[valid, 2],
        mode="markers", marker=dict(size=1.5, color="#00f0ff", opacity=0.4),
        hoverinfo="skip", name=f"{valid.sum()} tracked", showlegend=False)


def compute_dynamic(time_offset_min=0):
    now = datetime.now(timezone.utc) + timedelta(minutes=time_offset_min)
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
                             rel_v=float(np.linalg.norm(va[idx] - vb[idx])), pa=pa[idx], pb=pb[idx], tca_idx=idx))

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

    leaderboard = []
    for r in top_risks:
        age_a = tle_age_hours(SATS[r["i"]], jd0, fr0)
        age_b = tle_age_hours(SATS[r["j"]], jd0, fr0)
        sigma = combined_sigma(age_a, age_b)
        pc = probability_of_collision(r["miss_km"] * 1000, sigma, HARD_BODY_RADIUS_M)
        tca_seconds_from_now = r["tca_idx"] * FINE_STEP_MIN * 60
        lead_s = min(90 * 60, tca_seconds_from_now)
        burn_time = now + timedelta(seconds=(tca_seconds_from_now - lead_s))
        jd_b, fr_b = jday(burn_time.year, burn_time.month, burn_time.day,
                           burn_time.hour, burn_time.minute, burn_time.second + burn_time.microsecond / 1e6)
        _, r0_burn, v0_burn = SATS[r["i"]].sgp4(jd_b, fr_b)
        tag, tag_color = severity_tag(pc)
        leaderboard.append(dict(
            a=r["a"], b=r["b"], pair=f"{r['a']} / {r['b']}", miss_km=round(r["miss_km"], 2),
            rel_v=round(r["rel_v"], 2), pc=f"{pc:.2e}", pc_raw=pc, risk=round(r["risk"], 1), severity=tag,
            tca_min=round(tca_seconds_from_now / 60, 1),
            r0=list(r0_burn), v0=list(v0_burn), r_rel_km=(r["pa"] - r["pb"]).tolist(),
            t_flight_s=lead_s, mid=((r["pa"] + r["pb"]) / 2).tolist(),
            pa=r["pa"].tolist(), pb=r["pb"].tolist(),
        ))

    risk_slots = []
    for r, lb in zip(top_risks, leaderboard):
        label = f"{lb['pair']}<br>Miss: {lb['miss_km']} km<br>Pc: {lb['pc']}<br>{lb['severity']}"
        color = "#ef4444" if lb["severity"] == "CRITICAL" else ("#f59e0b" if lb["severity"] == "MODERATE" else "#10b981")
        risk_slots.append(dict(x=[r["pa"][0], r["pb"][0]], y=[r["pa"][1], r["pb"][1]], z=[r["pa"][2], r["pb"][2]],
                                color=color, text=[label, label], name=lb["severity"]))
    while len(risk_slots) < MAX_RISK_SLOTS:
        risk_slots.append(dict(x=[], y=[], z=[], color="#ef4444", text=[], name=""))

    stats = dict(count=int(valid.sum()), flagged=len([l for l in leaderboard if l["severity"] != "NOMINAL"]),
        closest=min((l["miss_km"] for l in leaderboard), default=float("nan")),
        max_pc=max((l["pc_raw"] for l in leaderboard), default=0.0),
        updated=now.strftime("%Y-%m-%d %H:%M:%S UTC"), leaderboard=leaderboard)
    return current_pos, valid, risk_slots, stats, now, jd0, fr0


_static = build_static_traces()
_init_pos, _init_valid, _init_risk_slots, _init_stats, _init_now, _, _ = compute_dynamic(0)
_lock_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=8, color="#ffffff", symbol="diamond",
    line=dict(color="#00f0ff", width=2)), hoverinfo="text", text=[], name="Selected", showlegend=False)
_orbit_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#00f0ff", width=3, dash="dot"),
    hoverinfo="skip", name="Orbit", showlegend=False)
_initial_fig = go.Figure(data=_static + [build_swarm_trace(_init_pos, _init_valid)] + [
    go.Scatter3d(x=s["x"], y=s["y"], z=s["z"], mode="lines+markers", line=dict(color=s["color"], width=6),
        marker=dict(size=4, color=s["color"]), text=s["text"], hoverinfo="text", name=s["name"], showlegend=False)
    for s in _init_risk_slots] + [_lock_trace, _orbit_trace])
_initial_fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               bgcolor="#080c14", aspectmode="data",
               camera=dict(eye=dict(x=1.15, y=1.15, z=0.7), center=dict(x=0, y=0, z=0))),
    paper_bgcolor="#080c14", font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
    showlegend=False, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep")

# ---------------------------------------------------------------- UI helpers
def stat_card(label, value_id, color="#00f0ff"):
    return html.Div(className="card", children=[
        html.Div(label, className="card-label"),
        html.Div("--", id=value_id, className="card-value mono", style={"color": color}),
    ])


LEADERBOARD_COLUMNS = [
    {"name": "Pair", "id": "pair"}, {"name": "Miss (km)", "id": "miss_km"},
    {"name": "RelV (km/s)", "id": "rel_v"}, {"name": "Pc", "id": "pc"}, {"name": "Sev", "id": "severity"},
]

app = Dash(__name__)
app.layout = html.Div(className="app-shell", children=[

    html.Div(className="topbar", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "0 20px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center"}, children=[
            html.Span(className="live-dot"),
            html.Span("ORBITSENSE", style={"fontWeight": 700, "fontSize": "15px", "color": "#e2e8f0", "letterSpacing": "1px"}),
            html.Span(" // SSA-OPS", className="mono", style={"color": "#64748b", "fontSize": "12px", "marginLeft": "4px"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
            dcc.Input(id="search-input", type="text", placeholder="Search satellite or NORAD ID...",
                      style={"width": "260px"}, n_submit=0),
            html.Button("Focus Camera", id="search-btn", n_clicks=0, className="btn btn-primary"),
            html.Div(id="search-feedback", className="mono", style={"color": "#f59e0b", "fontSize": "11px"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
            dcc.RadioItems(id="view-mode",
                options=[{"label": " STUDENT", "value": "student"}, {"label": " ADVANCED", "value": "advanced"}],
                value="student", className="dash-radio-items mono", inline=True),
            html.Div(id="utc-clock", className="mono", style={"color": "#00f0ff", "fontSize": "12px"}),
        ]),
    ]),

    html.Div(className="left-deck", children=[
        html.Div("FLEET STATUS", className="section-title"),
        stat_card("Tracked Objects", "stat-count"),
        stat_card("Flagged Conjunctions", "stat-flagged", color="#f59e0b"),
        stat_card("Closest Approach", "stat-closest"),
        stat_card("Highest Pc", "stat-pc", color="#ef4444"),
        html.Div(id="mode-status", className="mono", style={"color": "#10b981", "fontSize": "11px", "marginTop": "8px"}),
    ]),

    html.Div(className="center-canvas", children=[
        dcc.Graph(id="live-graph", figure=_initial_fig, style={"height": "100%", "width": "100%"}, config={"displayModeBar": False}),
        html.Div(style={"position": "absolute", "bottom": "12px", "right": "12px",
                         "background": "rgba(13,21,36,0.85)", "border": "1px solid #1e293b", "borderRadius": "8px",
                         "padding": "10px 14px", "fontSize": "10.5px", "lineHeight": "1.8"}, className="mono", children=[
            html.Div("LEGEND", style={"color": "#64748b", "fontWeight": 700, "marginBottom": "4px", "letterSpacing": "1px"}),
            html.Div([html.Span("\u25cf", style={"color": "#00f0ff", "marginRight": "6px"}), "Active"]),
            html.Div([html.Span("\u25cf", style={"color": "#f59e0b", "marginRight": "6px"}), "Moderate Risk"]),
            html.Div([html.Span("\u25cf", style={"color": "#ef4444", "marginRight": "6px"}), "Severe Conjunction"]),
            html.Div([html.Span("\u25c6", style={"color": "#ffffff", "marginRight": "6px"}), "Selected"]),
        ]),
    ]),

    html.Div(className="right-deck", children=[
        html.Div("THREAT LEADERBOARD", className="section-title"),
        html.Div(className="threat-table", children=[
            dash_table.DataTable(
                id="threat-table", columns=LEADERBOARD_COLUMNS, data=[],
                row_selectable=False, cell_selectable=True, page_size=10,
                style_table={"overflowX": "hidden"},
                style_cell={"textAlign": "left", "padding": "6px 8px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{severity} = "CRITICAL"'}, "color": "#ef4444"},
                    {"if": {"filter_query": '{severity} = "MODERATE"'}, "color": "#f59e0b"},
                ],
            ),
        ]),
        html.Div("SELECTED CONJUNCTION", className="section-title", style={"marginTop": "18px"}),
        html.Div(id="selected-telemetry", className="card mono", style={"fontSize": "11.5px", "lineHeight": "1.8"}),

        html.Div(id="student-panel", children=[html.Div(id="student-output")]),
        html.Div(id="advanced-panel", style={"display": "none"}, children=[
            html.Div("SANDBOX \u2014 \u0394v MANEUVER", className="section-title", style={"marginTop": "14px"}),
            dcc.RadioItems(id="mode-toggle",
                options=[{"label": " Physics", "value": "physics"}, {"label": " AI", "value": "ai"}],
                value="physics", className="dash-radio-items mono", style={"marginBottom": "10px"}, inline=True),
            html.Div("\u0394v (m/s)", className="mono", style={"fontSize": "10.5px", "color": "#64748b"}),
            dcc.Slider(id="dv-slider", min=-5, max=5, step=0.1, value=0, marks={-5: "-5", 0: "0", 5: "+5"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div("\u03c3 tracking uncertainty (m)", className="mono", style={"fontSize": "10.5px", "color": "#64748b", "marginTop": "10px"}),
            dcc.Slider(id="sigma-slider", min=100, max=2000, step=50, value=1000,
                       marks={100: "100", 1000: "1000", 2000: "2000"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(id="sandbox-output", className="mono", style={"marginTop": "12px", "fontSize": "11.5px", "lineHeight": "1.8"}),
        ]),
    ]),

    html.Div(className="bottom-dock", children=[
        html.Div(id="time-label", className="mono", style={"fontSize": "11px", "color": "#64748b", "marginBottom": "4px"}),
        dcc.Slider(id="time-slider", min=-TIME_SCRUB_LIMIT_MIN, max=TIME_SCRUB_LIMIT_MIN, step=1, value=0,
                   marks={-180: "-3h", -90: "-1.5h", 0: "NOW", 90: "+1.5h", 180: "+3h"},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "8px", "flexWrap": "wrap"}, children=[
            html.Button("|\u25c0\u25c0 -5m", id="step-back-btn", n_clicks=0, className="btn"),
            html.Button("\u25b6 Play", id="play-btn", n_clicks=0, className="btn btn-primary"),
            html.Button("+5m \u25b6\u25b6|", id="step-fwd-btn", n_clicks=0, className="btn"),
            html.Button("\u21ba Reset Now", id="reset-btn", n_clicks=0, className="btn btn-amber"),
            dcc.Dropdown(id="speed-choice", options=[{"label": "1x", "value": 1}, {"label": "5x", "value": 5},
                         {"label": "10x", "value": 10}, {"label": "50x", "value": 50}],
                         value=10, clearable=False, style={"width": "80px"}),
            html.Button("\u23f8 Pause Live", id="pause-btn", n_clicks=0, className="btn", style={"marginLeft": "auto"}),
            dcc.Dropdown(id="interval-choice", options=[{"label": "15s", "value": 15}, {"label": "30s", "value": 30},
                         {"label": "60s", "value": 60}], value=30, clearable=False, style={"width": "80px"}),
        ]),
    ]),

    dcc.Interval(id="refresh", interval=30 * 1000, n_intervals=0),
    dcc.Interval(id="play-interval", interval=2000, n_intervals=0, disabled=True),
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
    dcc.Store(id="leaderboard-store", data=_init_stats["leaderboard"]),
    dcc.Store(id="selected-pair-store", data=(_init_stats["leaderboard"][0] if _init_stats["leaderboard"] else None)),
    dcc.Store(id="focus-camera-store", data=None),
    dcc.Store(id="locked-sat-store", data=None),
])


@app.callback(Output("utc-clock", "children"), Input("clock-interval", "n_intervals"))
def tick_clock(n):
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


@app.callback(Output("pause-btn", "children"), Input("pause-btn", "n_clicks"))
def pause_label(n_clicks):
    return "\u25b6 Resume Live" if n_clicks % 2 == 1 else "\u23f8 Pause Live"


@app.callback(Output("refresh", "disabled"), Input("pause-btn", "n_clicks"), Input("time-slider", "value"))
def refresh_disabled(n_clicks, time_offset):
    return (n_clicks % 2 == 1) or (time_offset != 0)


@app.callback(Output("refresh", "interval"), Input("interval-choice", "value"))
def set_interval(seconds):
    return seconds * 1000


@app.callback(Output("student-panel", "style"), Output("advanced-panel", "style"), Output("mode-status", "children"),
              Input("view-mode", "value"))
def toggle_view_mode(mode):
    if mode == "student":
        return {"display": "block"}, {"display": "none"}, "\u25cf MODE: STUDENT"
    return {"display": "none"}, {"display": "block"}, "\u25cf MODE: ADVANCED"


@app.callback(
    Output("locked-sat-store", "data"), Output("search-feedback", "children"),
    Input("search-btn", "n_clicks"), Input("search-input", "n_submit"), State("search-input", "value"),
)
def do_search(n_clicks, n_submit, query):
    if not query:
        return None, ""
    idx, name = find_satellite(query)
    if idx is None:
        return None, f'No match: "{query}"'
    return idx, f"\u2713 {name}"


@app.callback(
    Output("selected-pair-store", "data"), Output("focus-camera-store", "data"),
    Input("threat-table", "active_cell"), Input("leaderboard-store", "data"),
    State("selected-pair-store", "data"),
)
def select_from_table(active_cell, leaderboard, previous_selection):
    if not leaderboard:
        return None, no_update

    # A real click always wins, uses fresh data, and moves the camera to that pair's midpoint.
    if ctx.triggered_id == "threat-table" and active_cell:
        row = active_cell["row"]
        if row < len(leaderboard):
            entry = leaderboard[row]
            return entry, entry["mid"]

    # Otherwise the leaderboard just refreshed: keep watching the SAME pair (matched by
    # identity, not row index -- the table re-sorts by risk every refresh, so a fixed
    # row index would silently point at a different pair after re-sorting).
    if previous_selection is not None:
        for entry in leaderboard:
            if entry["a"] == previous_selection["a"] and entry["b"] == previous_selection["b"]:
                return entry, no_update
        return leaderboard[0], no_update  # that pair is no longer flagged -- fall back to new top

    return leaderboard[0], no_update


@app.callback(Output("play-btn", "children"), Output("play-interval", "disabled"), Input("play-btn", "n_clicks"))
def toggle_play(n_clicks):
    playing = n_clicks % 2 == 1
    return ("\u23f8 Pause" if playing else "\u25b6 Play"), (not playing)


@app.callback(
    Output("time-slider", "value"),
    Input("play-interval", "n_intervals"), Input("step-fwd-btn", "n_clicks"),
    Input("step-back-btn", "n_clicks"), Input("reset-btn", "n_clicks"),
    State("time-slider", "value"), State("speed-choice", "value"),
    prevent_initial_call=True,
)
def move_time(play_ticks, fwd_clicks, back_clicks, reset_clicks, current, speed):
    trig = ctx.triggered_id
    if trig == "reset-btn":
        return 0
    if trig == "step-fwd-btn":
        return min(current + 5, TIME_SCRUB_LIMIT_MIN)
    if trig == "step-back-btn":
        return max(current - 5, -TIME_SCRUB_LIMIT_MIN)
    if trig == "play-interval":
        return min(current + speed, TIME_SCRUB_LIMIT_MIN)
    return current


@app.callback(
    Output("live-graph", "figure"), Output("stat-count", "children"), Output("stat-flagged", "children"),
    Output("stat-closest", "children"), Output("stat-pc", "children"), Output("time-label", "children"),
    Output("leaderboard-store", "data"), Output("threat-table", "data"),
    Input("refresh", "n_intervals"), Input("locked-sat-store", "data"),
    Input("time-slider", "value"), Input("focus-camera-store", "data"),
)
def update(n, locked_idx, time_offset, focus_target):
    current_pos, valid, risk_slots, stats, now, jd0, fr0 = compute_dynamic(time_offset)
    lb = stats["leaderboard"]
    table_data = [{"pair": l["pair"], "miss_km": l["miss_km"], "rel_v": l["rel_v"], "pc": l["pc"], "severity": l["severity"]} for l in lb]

    time_label = f"VIEWING: {stats['updated']}" + ("" if time_offset == 0 else f"  ({time_offset:+d} min)")
    closest_txt = f"{stats['closest']:.2f} km" if stats['closest'] == stats['closest'] else "--"
    pc_txt = f"{stats['max_pc']:.2e}"

    lock_x, lock_y, lock_z, lock_text = [], [], [], []
    orbit_x, orbit_y, orbit_z = [], [], []
    if locked_idx is not None:
        t = satellite_telemetry(locked_idx, jd0, fr0)
        lock_x, lock_y, lock_z = [t["pos"][0]], [t["pos"][1]], [t["pos"][2]]
        lock_text = [f"{t['name']} (NORAD {t['norad_id']})"]
        path = satellite_orbit_path(locked_idx, now)
        if len(path):
            orbit_x, orbit_y, orbit_z = path[:, 0].tolist(), path[:, 1].tolist(), path[:, 2].tolist()

    is_first_load = ctx.triggered_id is None
    is_new_lock = ctx.triggered_id == "locked-sat-store"
    is_new_focus = ctx.triggered_id == "focus-camera-store" and focus_target is not None

    if is_first_load:
        return _initial_fig, f"{stats['count']:,}", str(stats['flagged']), closest_txt, pc_txt, time_label, lb, table_data

    patched = Patch()
    patched["data"][3]["x"] = current_pos[valid, 0]
    patched["data"][3]["y"] = current_pos[valid, 1]
    patched["data"][3]["z"] = current_pos[valid, 2]
    for idx, s in enumerate(risk_slots):
        ti = 4 + idx
        patched["data"][ti]["x"] = s["x"]; patched["data"][ti]["y"] = s["y"]; patched["data"][ti]["z"] = s["z"]
        patched["data"][ti]["line"]["color"] = s["color"]; patched["data"][ti]["marker"]["color"] = s["color"]
        patched["data"][ti]["text"] = s["text"]; patched["data"][ti]["name"] = s["name"]
    patched["data"][LOCK_TRACE_IDX]["x"] = lock_x
    patched["data"][LOCK_TRACE_IDX]["y"] = lock_y
    patched["data"][LOCK_TRACE_IDX]["z"] = lock_z
    patched["data"][LOCK_TRACE_IDX]["text"] = lock_text
    patched["data"][ORBIT_TRACE_IDX]["x"] = orbit_x
    patched["data"][ORBIT_TRACE_IDX]["y"] = orbit_y
    patched["data"][ORBIT_TRACE_IDX]["z"] = orbit_z

    if is_new_lock and locked_idx is not None:
        pos = np.array([lock_x[0], lock_y[0], lock_z[0]])
        direction = pos / np.linalg.norm(pos)
        eye, center = direction * 0.35, direction * 0.28
        patched["layout"]["scene"]["camera"] = dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                                                      center=dict(x=center[0], y=center[1], z=center[2]))
    elif is_new_focus:
        pos = np.array(focus_target)
        direction = pos / np.linalg.norm(pos)
        eye, center = direction * 0.4, direction * 0.3
        patched["layout"]["scene"]["camera"] = dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                                                      center=dict(x=center[0], y=center[1], z=center[2]))

    return patched, f"{stats['count']:,}", str(stats['flagged']), closest_txt, pc_txt, time_label, lb, table_data


@app.callback(Output("selected-telemetry", "children"), Input("selected-pair-store", "data"))
def update_selected_telemetry(pair):
    if not pair:
        return "No conjunction flagged yet."
    return html.Div([
        html.Div(pair["pair"], style={"color": "#e2e8f0", "fontWeight": 700, "marginBottom": "6px"}),
        html.Div(f"TCA: T-{pair['tca_min']:.1f} min"),
        html.Div(f"Miss: {pair['miss_km']} km   RelV: {pair['rel_v']} km/s"),
        html.Div(f"Pc: {pair['pc']}"),
        html.Div(pair["severity"], style={"color": "#ef4444" if pair["severity"] == "CRITICAL" else ("#f59e0b" if pair["severity"] == "MODERATE" else "#10b981"),
                                            "fontWeight": 700, "marginTop": "6px"}),
    ])


@app.callback(Output("student-output", "children"), Input("selected-pair-store", "data"))
def update_student_panel(pair):
    if not pair:
        return ""
    level, color, explanation = plain_language_risk(pair["pc_raw"])
    return html.Div(className="card", children=[
        html.Div(f"Risk level: {level}", style={"color": color, "fontWeight": "bold", "fontSize": "15px", "marginBottom": "6px"}),
        html.Div(explanation, style={"fontSize": "11.5px", "lineHeight": "1.6", "opacity": 0.9}),
        html.Div("Switch to Advanced mode to run what-if maneuvers.",
                 style={"fontSize": "10px", "opacity": 0.55, "marginTop": "10px", "fontStyle": "italic"}),
    ])


@app.callback(
    Output("sandbox-output", "children"),
    Input("dv-slider", "value"), Input("sigma-slider", "value"),
    Input("mode-toggle", "value"), Input("selected-pair-store", "data"),
)
def update_sandbox(dv_m_s, sigma_m, mode, pair):
    if not pair:
        return ""
    lead_min = pair["t_flight_s"] / 60
    low_lead_note = (
        html.Div(f"\u26a0 Only {lead_min:.0f} min lead time available.",
                 style={"color": "#f59e0b", "fontSize": "10px", "marginBottom": "8px"})
        if lead_min < 15 else None
    )
    if mode == "ai":
        return html.Div([
            html.Div(f"Isolation Forest anomaly rank: {pair['risk']:.0f} / 100", style={"color": "#00f0ff"}),
            html.Div("(population-relative — doesn't move with these sliders.)", style={"opacity": 0.6, "fontSize": "10px", "marginTop": "6px"}),
        ])
    r0 = np.array(pair["r0"]); v0 = np.array(pair["v0"])
    r_rel_baseline = np.array(pair["r_rel_km"])
    R_hat = r0 / np.linalg.norm(r0)
    W_hat = np.cross(r0, v0); W_hat /= np.linalg.norm(W_hat)
    S_hat = np.cross(W_hat, R_hat)
    n = np.sqrt(MU_EARTH_KM3S2 / np.linalg.norm(r0) ** 3)
    dv_rsw = np.array([0, dv_m_s / 1000.0, 0])
    delta_r_rsw = cw_delta_r(n, pair["t_flight_s"], dv_rsw)
    delta_r_eci = delta_r_rsw[0] * R_hat + delta_r_rsw[1] * S_hat + delta_r_rsw[2] * W_hat
    new_miss_km = np.linalg.norm(r_rel_baseline + delta_r_eci)
    pc = probability_of_collision(new_miss_km * 1000, sigma_m, HARD_BODY_RADIUS_M)
    rec_text, rec_color = action_recommendation(pc)
    return html.Div([
        low_lead_note,
        html.Div(f"Miss distance: {new_miss_km * 1000:.1f} m", style={"color": "#00f0ff"}),
        html.Div(f"Pc: {pc:.3e}", style={"color": "#00f0ff", "marginTop": "2px"}),
        html.Div(rec_text, style={"color": rec_color, "fontWeight": "bold", "marginTop": "8px"}),
    ])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
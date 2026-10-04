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
CAMERA_CENTER_SCALE_KM = 18000.0
CACHE_FILE = os.path.join(os.path.dirname(__file__), "tle_cache.txt")
CACHE_MAX_AGE_HOURS = 2

COL_BLUE = "#4f8fc0"
COL_AMBER = "#c98a2e"
COL_RED = "#c14b4b"
COL_GREEN = "#3d9469"
COL_HIGHLIGHT = "#f2f2f2"
DEFAULT_CAMERA = dict(
    eye=dict(x=1.45, y=1.35, z=1.2),
    center=dict(x=0.0, y=0.0, z=0.0),
    up=dict(x=0.0, y=0.0, z=1.0),
)

TIMEZONE_OPTIONS = [
    {"label": "UTC", "value": 0.0},
    {"label": "IST (UTC+5:30)", "value": 5.5},
    {"label": "CET (UTC+1)", "value": 1.0},
    {"label": "EST (UTC-5)", "value": -5.0},
    {"label": "PST (UTC-8)", "value": -8.0},
    {"label": "JST (UTC+9)", "value": 9.0},
]

EARTH_TEXTURE_URL = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg"
GRID_W, GRID_H = 140, 70
N_COLORS = 180
EARTH_INDEX_GRID = np.zeros((GRID_H, GRID_W), dtype=float)
EARTH_COLORSCALE = [[0, "#1b3a6b"], [1, "#2f73bf"]]
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

NAMES, SATS, NAMES_LOWER = [], [], []
_runtime_initialized = False


def fetch_tle_text():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    if os.path.exists(CACHE_FILE):
        age_h = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600
        if age_h < CACHE_MAX_AGE_HOURS:
            print(f"Using cached TLE data ({age_h:.2f}h old)...")
            with open(CACHE_FILE) as f:
                return f.read()
    print("Fetching fresh active-satellite catalog from CelesTrak...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        text = resp.text
        lines = text.strip().splitlines()
        if len(lines) < 3:
            raise RuntimeError(f"TLE fetch returned too few lines ({len(lines)}).")
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except (requests.RequestException, RuntimeError) as exc:
        if os.path.exists(CACHE_FILE):
            print(f"WARNING: failed to fetch fresh TLE data ({exc}); using cached catalog.")
            with open(CACHE_FILE, encoding="utf-8") as f:
                return f.read()
        raise RuntimeError(
            "Unable to download TLE catalog from CelesTrak and no cache is available. "
            "Check your network and retry."
        ) from exc


def load_earth_texture():
    global EARTH_INDEX_GRID, EARTH_COLORSCALE
    print("Building textured Earth...")
    for attempt in range(1, 4):
        try:
            resp = requests.get(EARTH_TEXTURE_URL, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            img_small = img.resize((GRID_W, GRID_H), Image.LANCZOS).convert("RGB")
            img_q = img_small.quantize(colors=N_COLORS, method=Image.MEDIANCUT)
            _palette = img_q.getpalette()[: N_COLORS * 3]
            _palette_rgb = [tuple(_palette[i : i + 3]) for i in range(0, len(_palette), 3)]
            EARTH_INDEX_GRID = np.array(img_q).astype(float)
            EARTH_COLORSCALE = [
                [i / (N_COLORS - 1), f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(_palette_rgb)
            ]
            print("Earth ready.")
            return
        except Exception as exc:
            print(f"WARNING: Earth texture load attempt {attempt}/3 failed: {exc}")
            time.sleep(0.5)
    print("WARNING: using fallback procedural Earth texture.")
    lat_gradient = np.linspace(0, 1, GRID_H).reshape(-1, 1)
    EARTH_INDEX_GRID = np.repeat(lat_gradient, GRID_W, axis=1)
    EARTH_COLORSCALE = [[0, "#12355b"], [0.45, "#225f9b"], [1, "#8bb7d9"]]


def initialize_runtime(force=False):
    global NAMES, SATS, NAMES_LOWER, _runtime_initialized
    if _runtime_initialized and not force:
        return
    load_earth_texture()
    print("Loading satellite catalog...")
    raw_text = fetch_tle_text()
    lines = raw_text.strip().splitlines()
    names, sats = [], []
    for i in range(0, len(lines) - 2, 3):
        names.append(lines[i].strip())
        sats.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
    if len(sats) == 0:
        raise RuntimeError("Parsed 0 satellites; check TLE cache/content.")
    NAMES, SATS = names, sats
    NAMES_LOWER = [n.lower() for n in NAMES]
    _runtime_initialized = True
    print(f"Loaded {len(SATS)} satellites.")


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
        return "CRITICAL", COL_RED
    if pc > 1e-6:
        return "MODERATE", COL_AMBER
    return "NOMINAL", COL_GREEN


def action_recommendation(pc):
    if pc > 1e-4:
        return "CRITICAL: Execute Avoidance Maneuver", COL_RED
    if pc > 1e-6:
        return "WARNING: Monitor TLE", COL_AMBER
    return "NOMINAL: Orbit Clear", COL_GREEN


def plain_language_risk(pc):
    if pc > 1e-4:
        return "HIGH", COL_RED, "These two objects are close enough, and moving fast enough, that a collision is a real possibility."
    if pc > 1e-6:
        return "MODERATE", COL_AMBER, "These two objects will pass close to each other, worth continuing to track."
    return "LOW", COL_GREEN, "These two objects will miss each other by a comfortable margin."


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


def normalize_search_query(query):
    return (query or "").strip()


def search_satellite_lock(triggered_id, query):
    normalized = normalize_search_query(query)
    if triggered_id == "clear-focus-btn":
        return None, None, "", ""
    if not normalized:
        return None, None, "Enter a satellite name or NORAD ID.", query
    idx, name = find_satellite(normalized)
    if idx is None:
        return no_update, no_update, f'No match: "{normalized}"', query
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    sat_pos = satellite_telemetry(idx, jd, fr)["pos"]
    return idx, sat_pos.tolist(), f"LOCKED: {name} (NORAD {SATS[idx].satnum})", query


def satellite_telemetry(idx, jd, fr):
    sat = SATS[idx]
    err, pos, vel = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error for satellite index {idx}: code {err}")
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


def propagate_pair_series(sat_a, sat_b, jdfr_points):
    pa, pb, va, vb = [], [], [], []
    for jd, fr in jdfr_points:
        e1, p1, v1 = sat_a.sgp4(jd, fr)
        e2, p2, v2 = sat_b.sgp4(jd, fr)
        if e1 == 0 and e2 == 0:
            pa.append(p1)
            pb.append(p2)
            va.append(v1)
            vb.append(v2)
    if not pa:
        raise ValueError("No valid pair propagation points available.")
    return map(np.array, (pa, pb, va, vb))


def build_static_traces():
    rs = np.random.default_rng(42)
    n_stars = 800
    sr = rs.uniform(26000, STARFIELD_EXTENT, n_stars)
    st = rs.uniform(0, 2 * np.pi, n_stars)
    sp = rs.uniform(0, np.pi, n_stars)
    starfield = go.Scatter3d(x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
        mode="markers", marker=dict(size=1.1, color="#3a4250", opacity=0.6), hoverinfo="skip", showlegend=False)
    glow = go.Surface(x=GLOW_X, y=GLOW_Y, z=GLOW_Z, colorscale=[[0, COL_BLUE], [1, COL_BLUE]],
        showscale=False, opacity=0.08, hoverinfo="skip")
    earth = go.Surface(x=EARTH_X, y=EARTH_Y, z=EARTH_Z, surfacecolor=EARTH_INDEX_GRID, colorscale=EARTH_COLORSCALE,
        cmin=0, cmax=N_COLORS - 1, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.9))
    return [starfield, glow, earth]


def build_swarm_trace(current_pos, valid, highlight_idx=None):
    xs, ys, zs = current_pos[valid, 0], current_pos[valid, 1], current_pos[valid, 2]
    colors = [COL_BLUE] * len(xs)
    if highlight_idx is not None and valid[highlight_idx]:
        pos_in_swarm = int(np.searchsorted(np.where(valid)[0], highlight_idx))
        if 0 <= pos_in_swarm < len(colors):
            colors[pos_in_swarm] = COL_HIGHLIGHT
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=1.5, color=colors, opacity=0.55),
        hoverinfo="skip", name=f"{valid.sum()} tracked", showlegend=False)


def camera_for_target(target_pos):
    pos = np.asarray(target_pos, dtype=float)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        return None
    center = np.clip(-pos / CAMERA_CENTER_SCALE_KM, -0.6, 0.6)
    return dict(
        eye=DEFAULT_CAMERA["eye"],
        up=DEFAULT_CAMERA["up"],
        center=dict(x=float(center[0]), y=float(center[1]), z=float(center[2])),
    )


def build_dynamic_figure_patch(current_pos, valid, risk_slots, locked_idx, lock_xyz, lock_text, orbit_xyz, target_pos=None, reset_view=False):
    patched = Patch()
    swarm_colors = [COL_BLUE] * int(valid.sum())
    if locked_idx is not None and valid[locked_idx]:
        pos_in_swarm = int(np.searchsorted(np.where(valid)[0], locked_idx))
        if 0 <= pos_in_swarm < len(swarm_colors):
            swarm_colors[pos_in_swarm] = COL_HIGHLIGHT
    patched["data"][3]["x"] = current_pos[valid, 0]
    patched["data"][3]["y"] = current_pos[valid, 1]
    patched["data"][3]["z"] = current_pos[valid, 2]
    patched["data"][3]["marker"]["color"] = swarm_colors

    for idx, s in enumerate(risk_slots):
        ti = 4 + idx
        patched["data"][ti]["x"] = s["x"]; patched["data"][ti]["y"] = s["y"]; patched["data"][ti]["z"] = s["z"]
        patched["data"][ti]["line"]["color"] = s["color"]; patched["data"][ti]["marker"]["color"] = s["color"]
        patched["data"][ti]["text"] = s["text"]; patched["data"][ti]["name"] = s["name"]
    patched["data"][LOCK_TRACE_IDX]["x"] = lock_xyz[0]
    patched["data"][LOCK_TRACE_IDX]["y"] = lock_xyz[1]
    patched["data"][LOCK_TRACE_IDX]["z"] = lock_xyz[2]
    patched["data"][LOCK_TRACE_IDX]["text"] = lock_text
    patched["data"][ORBIT_TRACE_IDX]["x"] = orbit_xyz[0]
    patched["data"][ORBIT_TRACE_IDX]["y"] = orbit_xyz[1]
    patched["data"][ORBIT_TRACE_IDX]["z"] = orbit_xyz[2]

    if target_pos is not None:
        cam = camera_for_target(target_pos)
        if cam is not None:
            patched["layout"]["scene"]["camera"] = cam
    elif reset_view:
        patched["layout"]["scene"]["camera"] = DEFAULT_CAMERA
        patched["layout"]["scene"]["xaxis"]["autorange"] = True
        patched["layout"]["scene"]["yaxis"]["autorange"] = True
        patched["layout"]["scene"]["zaxis"]["autorange"] = True
    return patched


def compute_dynamic(time_offset_min=0):
    if not _runtime_initialized:
        initialize_runtime()
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
        pos_c.fill(np.nan)
        for bi, si in enumerate(idxs):
            for ti, (jd, fr) in enumerate(coarse_jdfr):
                err, pos, vel = SATS[si].sgp4(jd, fr)
                if err == 0:
                    pos_c[bi, ti] = pos
        min_dist = np.full((len(idxs), len(idxs)), np.inf)
        for t in range(len(coarse_times)):
            pt = pos_c[:, t, :]
            d = np.linalg.norm(pt[:, None, :] - pt[None, :, :], axis=-1)
            d[np.isnan(d)] = np.inf
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
        try:
            pa, pb, va, vb = propagate_pair_series(SATS[i], SATS[j], fine_jdfr)
        except ValueError:
            continue
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
        e_burn, r0_burn, v0_burn = SATS[r["i"]].sgp4(jd_b, fr_b)
        if e_burn != 0:
            continue
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
    for lb in leaderboard:
        label = f"{lb['pair']}<br>Miss: {lb['miss_km']} km<br>Pc: {lb['pc']}<br>{lb['severity']}"
        color = COL_RED if lb["severity"] == "CRITICAL" else (COL_AMBER if lb["severity"] == "MODERATE" else COL_GREEN)
        risk_slots.append(dict(x=[lb["pa"][0], lb["pb"][0]], y=[lb["pa"][1], lb["pb"][1]], z=[lb["pa"][2], lb["pb"][2]],
                                color=color, text=[label, label], name=lb["severity"]))
    while len(risk_slots) < MAX_RISK_SLOTS:
        risk_slots.append(dict(x=[], y=[], z=[], color=COL_RED, text=[], name=""))

    stats = dict(count=int(valid.sum()), flagged=len([l for l in leaderboard if l["severity"] != "NOMINAL"]),
        closest=min((l["miss_km"] for l in leaderboard), default=float("nan")),
        max_pc=max((l["pc_raw"] for l in leaderboard), default=0.0),
        updated_utc=now, leaderboard=leaderboard)
    return current_pos, valid, risk_slots, stats, now, jd0, fr0


_initial_fig = go.Figure()
_init_stats = {"leaderboard": [], "updated_utc": datetime.now(timezone.utc)}


def stat_card(label, value_id, color=COL_BLUE):
    return html.Div(className="card", children=[
        html.Div(label, className="card-label"),
        html.Div("--", id=value_id, className="card-value mono", style={"color": color}),
    ])


LEADERBOARD_COLUMNS = [
    {"name": "Pair", "id": "pair"}, {"name": "Miss (km)", "id": "miss_km"},
    {"name": "RelV (km/s)", "id": "rel_v"}, {"name": "Pc", "id": "pc"}, {"name": "Sev", "id": "severity"},
]

app = Dash(__name__)
app.layout = html.Div("OrbitSense is initializing...")


def build_initial_state():
    _static = build_static_traces()
    _init_pos, _init_valid, _init_risk_slots, _stats, _, _, _ = compute_dynamic(0)
    _lock_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=8, color=COL_HIGHLIGHT, symbol="diamond",
        line=dict(color=COL_BLUE, width=2)), hoverinfo="text", text=[], name="Selected", showlegend=False)
    _orbit_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color=COL_BLUE, width=3, dash="dot"),
        hoverinfo="skip", name="Orbit", showlegend=False)
    fig = go.Figure(data=_static + [build_swarm_trace(_init_pos, _init_valid)] + [
        go.Scatter3d(x=s["x"], y=s["y"], z=s["z"], mode="lines+markers", line=dict(color=s["color"], width=6),
            marker=dict(size=4, color=s["color"]), text=s["text"], hoverinfo="text", name=s["name"], showlegend=False)
        for s in _init_risk_slots] + [_lock_trace, _orbit_trace])
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   bgcolor="#151b22", aspectmode="data", camera=DEFAULT_CAMERA),
        paper_bgcolor="#151b22", font=dict(color="#dce3eb", family="Consolas, Menlo, Monaco, monospace"),
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep")
    return fig, _stats


def configure_layout(initial_fig, initial_stats):
    app.layout = html.Div(className="app-shell", children=[

    html.Div(className="topbar", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "0 18px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center"}, children=[
            html.Span(className="live-dot"),
            html.Span("ORBITSENSE", style={"fontWeight": 700, "fontSize": "14px", "color": "#1c242d", "letterSpacing": "0.5px"}),
            html.Span(" SSA-OPS", className="mono", style={"color": "#586572", "fontSize": "11px", "marginLeft": "4px"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
            dcc.Input(id="search-input", type="text", placeholder="Search satellite or NORAD ID",
                      style={"width": "230px"}, n_submit=0),
            html.Button("LOCATE", id="search-btn", n_clicks=0, className="btn btn-primary", title="Locate selected satellite"),
            html.Button("CLEAR", id="clear-focus-btn", n_clicks=0, className="btn", title="Clear selected satellite"),
            html.Div(id="search-feedback", className="mono", style={"color": COL_AMBER, "fontSize": "10.5px"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
            dcc.RadioItems(id="view-mode",
                options=[{"label": " STUDENT", "value": "student"}, {"label": " ADVANCED", "value": "advanced"}],
                value="student", className="dash-radio-items mono", inline=True),
            dcc.Dropdown(id="tz-choice", options=TIMEZONE_OPTIONS, value=5.5, clearable=False,
                         style={"width": "150px"}),
            html.Div(id="clock", className="mono", style={"color": COL_BLUE, "fontSize": "12px"}),
        ]),
    ]),

    html.Div(className="left-deck", children=[
        html.Div("FLEET STATUS", className="section-title"),
        stat_card("Tracked Objects", "stat-count"),
        stat_card("Flagged Conjunctions", "stat-flagged", color=COL_AMBER),
        stat_card("Closest Approach", "stat-closest"),
        stat_card("Highest Pc", "stat-pc", color=COL_RED),
        html.Div(id="mode-status", className="mono", style={"color": COL_GREEN, "fontSize": "10.5px", "marginTop": "6px"}),
    ]),

    html.Div(className="center-canvas", children=[
        dcc.Graph(id="live-graph", figure=initial_fig, style={"height": "100%", "width": "100%"}, config={"displayModeBar": False}),
        html.Div(style={"position": "absolute", "bottom": "12px", "right": "12px",
                         "background": "rgba(245,241,232,0.94)", "border": "1px solid #b8b1a5", "borderRadius": "1px",
                         "padding": "9px 12px", "fontSize": "10px", "lineHeight": "1.8"}, className="mono", children=[
            html.Div("LEGEND", style={"color": "#586572", "fontWeight": 600, "marginBottom": "4px", "letterSpacing": "0.5px"}),
            html.Div([html.Span("\u25a0", style={"color": COL_BLUE, "marginRight": "6px"}), "Tracked"]),
            html.Div([html.Span("\u25a0", style={"color": COL_AMBER, "marginRight": "6px"}), "Moderate risk"]),
            html.Div([html.Span("\u25a0", style={"color": COL_RED, "marginRight": "6px"}), "Severe conjunction"]),
            html.Div([html.Span("\u25c6", style={"color": COL_HIGHLIGHT, "marginRight": "6px"}), "Selected / locked"]),
        ]),
    ]),

    html.Div(className="right-deck", children=[
        html.Div("THREAT LEADERBOARD", className="section-title"),
        html.Div(className="threat-table", children=[
            dash_table.DataTable(
                id="threat-table", columns=LEADERBOARD_COLUMNS, data=[],
                row_selectable=False, cell_selectable=True, page_size=10,
                style_table={"overflowX": "hidden"},
                style_cell={"textAlign": "left", "padding": "5px 7px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{severity} = "CRITICAL"'}, "color": COL_RED},
                    {"if": {"filter_query": '{severity} = "MODERATE"'}, "color": COL_AMBER},
                ],
            ),
        ]),
        html.Div("SELECTED CONJUNCTION", className="section-title", style={"marginTop": "16px"}),
        html.Div(id="selected-telemetry", className="card mono", style={"fontSize": "11px", "lineHeight": "1.8"}),

        html.Div(id="student-panel", children=[html.Div(id="student-output")]),
        html.Div(id="advanced-panel", style={"display": "none"}, children=[
            html.Div("SANDBOX: dV MANEUVER", className="section-title", style={"marginTop": "12px"}),
            dcc.RadioItems(id="mode-toggle",
                options=[{"label": " Physics", "value": "physics"}, {"label": " AI", "value": "ai"}],
                value="physics", className="dash-radio-items mono", style={"marginBottom": "8px"}, inline=True),
            html.Div("dV (m/s)", className="mono", style={"fontSize": "10px", "color": "#586572"}),
            dcc.Slider(id="dv-slider", min=-5, max=5, step=0.1, value=0, marks={-5: "-5", 0: "0", 5: "+5"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div("sigma tracking uncertainty (m)", className="mono", style={"fontSize": "10px", "color": "#586572", "marginTop": "8px"}),
            dcc.Slider(id="sigma-slider", min=100, max=2000, step=50, value=1000,
                       marks={100: "100", 1000: "1000", 2000: "2000"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(id="sandbox-output", className="mono", style={"marginTop": "10px", "fontSize": "11px", "lineHeight": "1.8"}),
        ]),
    ]),

    html.Div(className="bottom-dock", children=[
        html.Div(id="time-label", className="mono", style={"fontSize": "10.5px", "color": "#586572", "marginBottom": "4px"}),
        dcc.Slider(id="time-slider", min=-TIME_SCRUB_LIMIT_MIN, max=TIME_SCRUB_LIMIT_MIN, step=1, value=0,
                   marks={-180: "-3h", -90: "-1.5h", 0: "NOW", 90: "+1.5h", 180: "+3h"},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "6px", "flexWrap": "wrap"}, children=[
            html.Button("-5m", id="step-back-btn", n_clicks=0, className="btn", title="Move simulation backward by 5 minutes"),
            html.Button("PLAY", id="play-btn", n_clicks=0, className="btn btn-primary", title="Play time scrubber"),
            html.Button("+5m", id="step-fwd-btn", n_clicks=0, className="btn", title="Move simulation forward by 5 minutes"),
            html.Button("RESET", id="reset-btn", n_clicks=0, className="btn btn-amber", title="Reset time to now"),
            dcc.Dropdown(id="speed-choice", options=[{"label": "1x", "value": 1}, {"label": "5x", "value": 5},
                         {"label": "10x", "value": 10}, {"label": "50x", "value": 50}],
                         value=10, clearable=False, style={"width": "72px"}),
            html.Button("PAUSE LIVE", id="pause-btn", n_clicks=0, className="btn", title="Pause or resume live refresh", style={"marginLeft": "auto"}),
            dcc.Dropdown(id="interval-choice", options=[{"label": "15s", "value": 15}, {"label": "30s", "value": 30},
                         {"label": "60s", "value": 60}], value=30, clearable=False, style={"width": "72px"}),
        ]),
    ]),

    dcc.Interval(id="refresh", interval=30 * 1000, n_intervals=0),
    dcc.Interval(id="play-interval", interval=2000, n_intervals=0, disabled=True),
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
    dcc.Store(id="leaderboard-store", data=initial_stats["leaderboard"]),
    dcc.Store(id="selected-pair-store", data=(initial_stats["leaderboard"][0] if initial_stats["leaderboard"] else None)),
    dcc.Store(id="focus-camera-store", data=None),
    dcc.Store(id="locked-sat-store", data=None),
    dcc.Store(id="last-computed-utc", data=initial_stats["updated_utc"].isoformat()),
    ])


@app.callback(Output("clock", "children"), Input("clock-interval", "n_intervals"), Input("tz-choice", "value"))
def tick_clock(n, tz_offset):
    local = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
    label = next((o["label"].split(" ")[0] for o in TIMEZONE_OPTIONS if o["value"] == tz_offset), "UTC")
    return f"{local.strftime('%H:%M:%S')} {label}"


@app.callback(Output("pause-btn", "children"), Input("pause-btn", "n_clicks"))
def pause_label(n_clicks):
    return "RESUME LIVE" if n_clicks % 2 == 1 else "PAUSE LIVE"


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
        return {"display": "block"}, {"display": "none"}, "MODE: STUDENT"
    return {"display": "none"}, {"display": "block"}, "MODE: ADVANCED"


@app.callback(
    Output("locked-sat-store", "data"),
    Output("focus-camera-store", "data", allow_duplicate=True),
    Output("search-feedback", "children"),
    Output("search-input", "value"),
    Input("search-btn", "n_clicks"), Input("search-input", "n_submit"), Input("clear-focus-btn", "n_clicks"),
    State("search-input", "value"),
    prevent_initial_call=True,
)
def do_search(n_clicks, n_submit, n_clear, query):
    return search_satellite_lock(ctx.triggered_id, query)


@app.callback(
    Output("selected-pair-store", "data"), Output("focus-camera-store", "data"),
    Input("threat-table", "active_cell"), Input("leaderboard-store", "data"),
    State("selected-pair-store", "data"),
)
def select_from_table(active_cell, leaderboard, previous_selection):
    if not leaderboard:
        return None, no_update
    if ctx.triggered_id == "threat-table" and active_cell:
        row = active_cell["row"]
        if row < len(leaderboard):
            entry = leaderboard[row]
            return entry, entry["mid"]
    if previous_selection is not None:
        for entry in leaderboard:
            if entry["a"] == previous_selection["a"] and entry["b"] == previous_selection["b"]:
                return entry, no_update
        return leaderboard[0], no_update
    return leaderboard[0], no_update


@app.callback(Output("play-btn", "children"), Output("play-interval", "disabled"), Input("play-btn", "n_clicks"))
def toggle_play(n_clicks):
    playing = n_clicks % 2 == 1
    return ("PAUSE" if playing else "PLAY"), (not playing)


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
    Output("stat-closest", "children"), Output("stat-pc", "children"), Output("last-computed-utc", "data"),
    Output("leaderboard-store", "data"), Output("threat-table", "data"),
    Input("refresh", "n_intervals"), Input("locked-sat-store", "data"),
    Input("time-slider", "value"), Input("focus-camera-store", "data"),
)
def update(n, locked_idx, time_offset, focus_target):
    current_pos, valid, risk_slots, stats, now, jd0, fr0 = compute_dynamic(time_offset)

    utc_iso = stats["updated_utc"].isoformat()
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
    is_cleared_lock = is_new_lock and locked_idx is None
    is_new_focus = ctx.triggered_id == "focus-camera-store" and focus_target is not None
    is_cleared_focus = ctx.triggered_id == "focus-camera-store" and focus_target is None

    if is_first_load:
        return _initial_fig, f"{stats['count']:,}", str(stats['flagged']), closest_txt, pc_txt, utc_iso, stats["leaderboard"], _table_rows(stats["leaderboard"])

    target_pos = None
    if is_new_lock and locked_idx is not None:
        target_pos = np.array([lock_x[0], lock_y[0], lock_z[0]])
    elif is_new_focus:
        target_pos = np.array(focus_target)
    patched = build_dynamic_figure_patch(
        current_pos=current_pos,
        valid=valid,
        risk_slots=risk_slots,
        locked_idx=locked_idx,
        lock_xyz=(lock_x, lock_y, lock_z),
        lock_text=lock_text,
        orbit_xyz=(orbit_x, orbit_y, orbit_z),
        target_pos=target_pos,
        reset_view=(is_cleared_lock or is_cleared_focus),
    )

    return patched, f"{stats['count']:,}", str(stats['flagged']), closest_txt, pc_txt, utc_iso, stats["leaderboard"], _table_rows(stats["leaderboard"])


def _table_rows(leaderboard):
    return [{"pair": l["pair"], "miss_km": l["miss_km"], "rel_v": l["rel_v"], "pc": l["pc"], "severity": l["severity"]} for l in leaderboard]


@app.callback(
    Output("time-label", "children"),
    Input("last-computed-utc", "data"), Input("tz-choice", "value"), Input("time-slider", "value"),
)
def format_time_label(utc_iso, tz_offset, time_offset):
    # Pure string formatting -- no physics recomputation. Changing timezone here never
    # re-triggers the expensive conjunction-detection pipeline.
    utc_dt = datetime.fromisoformat(utc_iso)
    display_time = utc_dt + timedelta(hours=tz_offset)
    tz_label = next((o["label"].split(" ")[0] for o in TIMEZONE_OPTIONS if o["value"] == tz_offset), "UTC")
    suffix = "" if time_offset == 0 else f"  ({time_offset:+d} min)"
    return f"VIEWING: {display_time.strftime('%Y-%m-%d %H:%M:%S')} {tz_label}{suffix}"


@app.callback(
    Output("selected-telemetry", "children"),
    Input("selected-pair-store", "data"),
    Input("locked-sat-store", "data"),
    Input("last-computed-utc", "data"),
)
def update_selected_telemetry(pair, locked_idx, utc_iso):
    sections = []
    if locked_idx is not None and utc_iso:
        dt = datetime.fromisoformat(utc_iso)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        try:
            sat = satellite_telemetry(locked_idx, jd, fr)
        except RuntimeError as exc:
            sections.append(html.Div(f"LOCK ERROR: {exc}", style={"color": COL_RED}))
        else:
            sections.append(html.Div([
                html.Div("LOCKED SATELLITE", className="section-title", style={"marginTop": "0"}),
                html.Div(sat["name"], style={"color": "#1c242d", "fontWeight": 600, "marginBottom": "4px"}),
                html.Div(f"NORAD: {sat['norad_id']}  Alt: {sat['altitude_km']:.1f} km"),
                html.Div(f"Speed: {sat['speed_kms']:.3f} km/s  Incl: {sat['inclination_deg']:.2f}\N{DEGREE SIGN}"),
            ], style={"marginBottom": "10px"}))
    if pair:
        sections.append(html.Div([
            html.Div("SELECTED CONJUNCTION", className="section-title", style={"marginTop": "0"}),
            html.Div(pair["pair"], style={"color": "#1c242d", "fontWeight": 600, "marginBottom": "6px"}),
            html.Div(f"TCA: T-{pair['tca_min']:.1f} min"),
            html.Div(f"Miss: {pair['miss_km']} km   RelV: {pair['rel_v']} km/s"),
            html.Div(f"Pc: {pair['pc']}"),
            html.Div(pair["severity"], style={"color": COL_RED if pair["severity"] == "CRITICAL" else (COL_AMBER if pair["severity"] == "MODERATE" else COL_GREEN),
                                                "fontWeight": 600, "marginTop": "6px"}),
        ]))
    if not sections:
        return "No conjunction flagged yet."
    return html.Div(sections)


@app.callback(Output("student-output", "children"), Input("selected-pair-store", "data"))
def update_student_panel(pair):
    if not pair:
        return ""
    level, color, explanation = plain_language_risk(pair["pc_raw"])
    return html.Div(className="card", children=[
        html.Div(f"Risk level: {level}", style={"color": color, "fontWeight": 600, "fontSize": "14px", "marginBottom": "6px"}),
        html.Div(explanation, style={"fontSize": "11px", "lineHeight": "1.6", "opacity": 0.9}),
        html.Div("Switch to Advanced mode to run what-if maneuvers.",
                 style={"fontSize": "9.5px", "opacity": 0.55, "marginTop": "8px", "fontStyle": "italic"}),
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
        html.Div(f"Only {lead_min:.0f} min lead time available.",
                 style={"color": COL_AMBER, "fontSize": "10px", "marginBottom": "6px"})
        if lead_min < 15 else None
    )
    if mode == "ai":
        return html.Div([
            html.Div(f"Isolation Forest anomaly rank: {pair['risk']:.0f} / 100", style={"color": COL_BLUE}),
            html.Div("(population-relative, not affected by these sliders.)", style={"opacity": 0.6, "fontSize": "10px", "marginTop": "6px"}),
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
        html.Div(f"Miss distance: {new_miss_km * 1000:.1f} m", style={"color": COL_BLUE}),
        html.Div(f"Pc: {pc:.3e}", style={"color": COL_BLUE, "marginTop": "2px"}),
        html.Div(rec_text, style={"color": rec_color, "fontWeight": 600, "marginTop": "8px"}),
    ])


def create_app():
    global _initial_fig, _init_stats
    initialize_runtime()
    _initial_fig, _init_stats = build_initial_state()
    configure_layout(_initial_fig, _init_stats)
    return app


if __name__ == "__main__":
    try:
        create_app().run(debug=False, port=8050)
    except Exception as exc:
        raise SystemExit(
            "OrbitSense failed to start. Check network connectivity for CelesTrak/texture downloads "
            f"or use a valid tle_cache.txt. Details: {exc}"
        ) from exc
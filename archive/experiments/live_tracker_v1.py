import requests
import numpy as np
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output

R_EARTH = 6371.0
BIN_WIDTH_KM = 25
MAX_PER_BUCKET = 400
COARSE_CANDIDATE_KM = 200
MAX_CANDIDATES = 300
WINDOW_MINUTES = 180
COARSE_STEP_MIN = 12
FINE_STEP_MIN = 1

# ---------- Build a realistic textured Earth ONCE at startup ----------
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

_gu, _gv = np.mgrid[0 : 2 * np.pi : 40j, 0:np.pi:20j]
GLOW_R = R_EARTH * 1.035
GLOW_X = GLOW_R * np.cos(_gu) * np.sin(_gv)
GLOW_Y = GLOW_R * np.sin(_gu) * np.sin(_gv)
GLOW_Z = GLOW_R * np.cos(_gv)

print("Earth ready.")

# ---------- Fetch satellite catalog ONCE at startup ----------
print("Fetching full active-satellite catalog from CelesTrak (one-time)...")
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
lines = requests.get(url).text.strip().splitlines()

NAMES, SATS = [], []
for i in range(0, len(lines) - 2, 3):
    NAMES.append(lines[i].strip())
    SATS.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
print(f"Loaded {len(SATS)} satellites.")


def compute_frame():
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
        refined.append(dict(a=NAMES[i], b=NAMES[j], miss_km=d[idx],
                             rel_v=float(np.linalg.norm(va[idx] - vb[idx])),
                             pa=pa[idx], pb=pb[idx]))

    if len(refined) >= 5:
        X = np.array([[r["miss_km"], r["rel_v"]] for r in refined])
        Xs = StandardScaler().fit_transform(X)
        model = IsolationForest(contamination=min(0.2, 8 / len(refined)), random_state=42).fit(Xs)
        anomaly = -model.score_samples(Xs)
        risk = 100 * (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)
        for r, s in zip(refined, risk):
            r["risk"] = s
        refined.sort(key=lambda r: -r["risk"])
    top_risks = refined[:10]

    fig = go.Figure()

    rs = np.random.default_rng(42)
    n_stars = 900
    sr = rs.uniform(32000, 48000, n_stars)
    st = rs.uniform(0, 2 * np.pi, n_stars)
    sp = rs.uniform(0, np.pi, n_stars)
    fig.add_trace(go.Scatter3d(
        x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
        mode="markers", marker=dict(size=1.2, color="white", opacity=0.55),
        hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Surface(
        x=GLOW_X, y=GLOW_Y, z=GLOW_Z,
        colorscale=[[0, "#3fa9ff"], [1, "#3fa9ff"]], showscale=False,
        opacity=0.12, hoverinfo="skip"))

    fig.add_trace(go.Surface(
        x=EARTH_X, y=EARTH_Y, z=EARTH_Z,
        surfacecolor=EARTH_INDEX_GRID, colorscale=EARTH_COLORSCALE,
        cmin=0, cmax=N_COLORS - 1, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.9)))

    fig.add_trace(go.Scatter3d(
        x=current_pos[valid, 0], y=current_pos[valid, 1], z=current_pos[valid, 2],
        mode="markers", marker=dict(size=1.6, color="#7fd7ff", opacity=0.45),
        hoverinfo="skip", name=f"{valid.sum()} tracked satellites", showlegend=True))

    for r in top_risks:
        label = f"{r['a']} vs {r['b']}<br>Miss: {r['miss_km']:.2f} km<br>Risk: {r['risk']:.0f}/100"
        color = "#ff3b3b" if r["risk"] > 60 else "#ffa63b"
        fig.add_trace(go.Scatter3d(
            x=[r["pa"][0], r["pb"][0]], y=[r["pa"][1], r["pb"][1]], z=[r["pa"][2], r["pb"][2]],
            mode="lines+markers", line=dict(color=color, width=7),
            marker=dict(size=5, color=color),
            text=[label, label], hoverinfo="text", showlegend=False))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   bgcolor="black", aspectmode="data"),
        paper_bgcolor="black", font=dict(color="#c9e8ff", family="Consolas, monospace"),
        showlegend=True, legend=dict(font=dict(color="#c9e8ff", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=0, b=0))

    stats = dict(
        count=int(valid.sum()),
        flagged=len([r for r in top_risks if r["risk"] > 60]),
        closest=min((r["miss_km"] for r in top_risks), default=float("nan")),
        updated=now.strftime("%H:%M:%S UTC"),
    )
    return fig, stats


HUD_STYLE = {
    "position": "absolute", "top": "18px", "left": "18px", "zIndex": 10,
    "backgroundColor": "rgba(5,10,20,0.75)", "border": "1px solid #2b6a99",
    "borderRadius": "6px", "padding": "14px 20px", "color": "#7fd7ff",
    "fontFamily": "Consolas, monospace", "fontSize": "14px", "lineHeight": "1.8",
    "boxShadow": "0 0 18px rgba(63,169,255,0.25)",
}

app = Dash(__name__)
app.layout = html.Div(style={"backgroundColor": "black", "height": "100vh", "position": "relative"}, children=[
    html.Div(id="hud", style=HUD_STYLE),
    dcc.Graph(id="live-graph", style={"height": "100vh"}, config={"displayModeBar": False}),
    dcc.Interval(id="refresh", interval=30 * 1000, n_intervals=0),
])


@app.callback(Output("live-graph", "figure"), Output("hud", "children"), Input("refresh", "n_intervals"))
def update(n):
    fig, stats = compute_frame()
    hud = [
        html.Div("ORBITSENSE — LIVE", style={"fontSize": "16px", "fontWeight": "bold", "color": "#fff", "marginBottom": "6px"}),
        html.Div(f"TRACKED OBJECTS ..... {stats['count']:,}"),
        html.Div(f"FLAGGED CONJUNCTIONS  {stats['flagged']}", style={"color": "#ff5c5c" if stats["flagged"] else "#7fd7ff"}),
        html.Div(f"CLOSEST APPROACH .... {stats['closest']:.2f} km" if stats['closest'] == stats['closest'] else "CLOSEST APPROACH .... --"),
        html.Div(f"LAST UPDATE ......... {stats['updated']}", style={"opacity": 0.7, "marginTop": "6px"}),
    ]
    return fig, hud


if __name__ == "__main__":
    app.run(debug=False, port=8050)
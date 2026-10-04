import requests
import itertools
import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------- 1. Pull a real, sizeable batch of satellites ----------
# Starlink is used deliberately: it's a huge constellation with many
# satellites in similar orbital shells, so genuine close approaches
# actually exist to find (unlike the 2-satellite "stations" group).
SAMPLE_SIZE = 150
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
try:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
except requests.RequestException as exc:
    raise RuntimeError(f"Failed to fetch Starlink TLE data from CelesTrak: {exc}") from exc
lines = resp.text.strip().splitlines()
if len(lines) < 3:
    raise RuntimeError("CelesTrak Starlink response did not contain enough TLE lines.")

satellites = []
for i in range(0, min(len(lines), SAMPLE_SIZE * 3), 3):
    name = lines[i].strip()
    sat = Satrec.twoline2rv(lines[i + 1], lines[i + 2])
    satellites.append((name, sat))

print(f"Loaded {len(satellites)} satellites")

# ---------- 2. Propagate every satellite across a time window ----------
WINDOW_MINUTES, STEP_MINUTES = 180, 2
now = datetime.now(timezone.utc)
times = [now + timedelta(minutes=m) for m in range(0, WINDOW_MINUTES, STEP_MINUTES)]
jds_frs = [jday(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in times]

all_pos, all_vel = {}, {}
for name, sat in satellites:
    pos_list, vel_list = [], []
    for jd, fr in jds_frs:
        err, pos, vel = sat.sgp4(jd, fr)
        pos_list.append(pos if err == 0 else (np.nan,) * 3)
        vel_list.append(vel if err == 0 else (np.nan,) * 3)
    all_pos[name] = np.array(pos_list)
    all_vel[name] = np.array(vel_list)

# ---------- 3. For every pair, find the real closest approach ----------
records = []
names = list(all_pos.keys())
for a, b in itertools.combinations(names, 2):
    pa, pb, va, vb = all_pos[a], all_pos[b], all_vel[a], all_vel[b]
    dists = np.linalg.norm(pa - pb, axis=1)
    idx = int(np.nanargmin(dists))
    records.append(dict(
        sat_a=a, sat_b=b,
        miss_distance_km=dists[idx],
        relative_velocity_kms=np.linalg.norm(va[idx] - vb[idx]),
        time_idx=idx,
    ))

print(f"Evaluated {len(records)} satellite pairs")

# ---------- 4. ML risk ranking (anomaly detection, not a fabricated classifier) ----------
# There's no real public dataset of "this conjunction became a collision" —
# thankfully, actual collisions are extremely rare. So instead of training on
# invented labels, this uses anomaly detection: it learns what's "normal" for
# this batch (typical miss distance / relative velocity) and flags pairs that
# are statistical outliers — which is exactly what a dangerously close,
# unusually fast pass looks like. The physics (steps 2-3) generates the
# features; the ML only ranks how unusual/risky they are.
X = np.array([[r["miss_distance_km"], r["relative_velocity_kms"]] for r in records])
X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)
anomaly = -model.score_samples(X_scaled)
risk_score = 100 * (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min())

for r, s in zip(records, risk_score):
    r["risk_score"] = s
records.sort(key=lambda r: -r["risk_score"])
top_risks = records[:8]

print("\nTop flagged conjunctions:")
for r in top_risks:
    print(f"  {r['sat_a']}  <->  {r['sat_b']}   "
          f"miss={r['miss_distance_km']:.2f} km   risk={r['risk_score']:.0f}/100")

# ---------- 5. Build the visual: swarm + highlighted risk ----------
fig = go.Figure()

np.random.seed(42)
n_stars = 600
sr = np.random.uniform(30000, 45000, n_stars)
st = np.random.uniform(0, 2 * np.pi, n_stars)
sp = np.random.uniform(0, np.pi, n_stars)
fig.add_trace(go.Scatter3d(
    x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
    mode="markers", marker=dict(size=1, color="white", opacity=0.6),
    hoverinfo="skip", showlegend=False))

R = 6371
u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
fig.add_trace(go.Surface(
    x=R * np.cos(u) * np.sin(v), y=R * np.sin(u) * np.sin(v), z=R * np.cos(v),
    colorscale=[[0, "#1b3a6b"], [1, "#1b3a6b"]], showscale=False, opacity=0.95, hoverinfo="skip"))

# The swarm: every tracked satellite's current position, quiet and small
current = np.array([all_pos[name][0] for name in names])
fig.add_trace(go.Scatter3d(
    x=current[:, 0], y=current[:, 1], z=current[:, 2], mode="markers",
    marker=dict(size=2, color="#5da8ff", opacity=0.45),
    text=names, hoverinfo="text", name="Tracked satellites"))

# The highlights: only the pairs that actually matter
for r in top_risks:
    pa = all_pos[r["sat_a"]][r["time_idx"]]
    pb = all_pos[r["sat_b"]][r["time_idx"]]
    label = (f"{r['sat_a']} vs {r['sat_b']}<br>"
              f"Miss distance: {r['miss_distance_km']:.2f} km<br>"
              f"Risk score: {r['risk_score']:.0f}/100")
    fig.add_trace(go.Scatter3d(
        x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]],
        mode="lines+markers", line=dict(color="#ff2b2b", width=6),
        marker=dict(size=4, color="#ff2b2b"),
        text=[label, label], hoverinfo="text",
        name=f"Risk {r['risk_score']:.0f}"))

fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               bgcolor="black", aspectmode="data"),
    paper_bgcolor="black", font=dict(color="white"),
    title=dict(text="OrbitSense — Conjunction Risk Map", font=dict(size=20, color="white")),
    legend=dict(font=dict(color="white", size=9)))

fig.write_html("risk_map.html")
print("\nDone! Open risk_map.html in your browser.")
import requests
import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone

# 1. Pull real, live satellite data — a small, recognizable set (space stations)
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
lines = requests.get(url).text.strip().splitlines()

satellites = []
for i in range(0, len(lines), 3):
    name = lines[i].strip()
    line1 = lines[i + 1]
    line2 = lines[i + 2]
    satellites.append((name, Satrec.twoline2rv(line1, line2)))

print("Found satellites:", [s[0] for s in satellites])

# 2. Propagate each satellite's path over roughly one full orbit
def get_orbit_path(sat, minutes=100, step=1):
    positions = []
    now = datetime.now(timezone.utc)
    for m in range(0, minutes, step):
        t = now + timedelta(minutes=m)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        err, pos, _ = sat.sgp4(jd, fr)
        if err == 0:
            positions.append(pos)
    return np.array(positions)

fig = go.Figure()

# 3. Starfield backdrop
np.random.seed(42)
n_stars = 800
star_r = np.random.uniform(30000, 45000, n_stars)
star_theta = np.random.uniform(0, 2 * np.pi, n_stars)
star_phi = np.random.uniform(0, np.pi, n_stars)
fig.add_trace(go.Scatter3d(
    x=star_r * np.cos(star_theta) * np.sin(star_phi),
    y=star_r * np.sin(star_theta) * np.sin(star_phi),
    z=star_r * np.cos(star_phi),
    mode="markers", marker=dict(size=1, color="white", opacity=0.7),
    hoverinfo="skip", showlegend=False))

# 4. Earth
R = 6371
u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
x = R * np.cos(u) * np.sin(v)
y = R * np.sin(u) * np.sin(v)
z = R * np.cos(v)
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, "#1b3a6b"], [1, "#1b3a6b"]],
                          showscale=False, opacity=0.95, hoverinfo="skip"))

# 5. Orbit paths, one color per satellite
colors = ["#ff4d4d", "#4dc3ff", "#ffd24d", "#4dff88", "#c94dff"]
for idx, (name, sat) in enumerate(satellites):
    path = get_orbit_path(sat)
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode="lines+markers",
        line=dict(color=colors[idx % len(colors)], width=5),
        marker=dict(size=2, color=colors[idx % len(colors)]),
        name=name))

# 6. Styling
fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               bgcolor="black", aspectmode="data"),
    paper_bgcolor="black", font=dict(color="white"),
    title=dict(text="OrbitSense — Live Orbital Tracks", font=dict(size=20, color="white")),
    legend=dict(font=dict(color="white")))

fig.write_html("orbit_view.html")
print("Done! Open orbit_view.html in your browser and drag to rotate.")
import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from scipy import integrate
from scipy.special import ive

# ============================================================================
# REAL historical TLE data for Iridium 33 and Cosmos 2251, published Feb 2009,
# just before the Feb 10, 2009 collision -- the first-ever accidental
# hypervelocity collision between two intact satellites.
# ============================================================================
IRIDIUM_L1 = "1 24946U 97051C   09040.78448243 +.00000153 +00000-0 +47668-4 0 04775"
IRIDIUM_L2 = "2 24946 086.3994 121.7028 0002288 085.1644 274.9812 14.34219863597336"
COSMOS_L1  = "1 22675U 93036A   09040.49834364 -.00000001 +00000-0 +95251-5 0 07411"
COSMOS_L2  = "2 22675 074.0355 019.4646 0016027 098.7014 261.5952 14.31135643817415"

DOCUMENTED_COLLISION_TIME = datetime(2009, 2, 10, 16, 55, 59, 800000, tzinfo=timezone.utc)
DOCUMENTED_REL_VELOCITY_KMS = 11.7   # widely reported, e.g. Wikipedia / peer-reviewed sources
DOCUMENTED_STK_MISS_M = 698.0        # independently computed via AGI STK 8 by a contemporaneous observer

R_EARTH = 6371.0


def probability_of_collision(miss_distance_m, sigma_m, hard_body_radius_m):
    """Standard 2D Pc for a circular combined-covariance encounter (Foster/Alfano-style)."""
    d, sig, R = miss_distance_m, sigma_m, hard_body_radius_m
    def integrand(r):
        z = r * d / sig ** 2
        exponent = -(r ** 2 + d ** 2) / (2 * sig ** 2) + z
        return (r / sig ** 2) * np.exp(exponent) * ive(0, z)
    pc, _ = integrate.quad(integrand, 0, R)
    return pc


def compute_historical_metrics(search_seconds=180, step_seconds=0.1):
    iridium = Satrec.twoline2rv(IRIDIUM_L1, IRIDIUM_L2)
    cosmos = Satrec.twoline2rv(COSMOS_L1, COSMOS_L2)
    best = None
    for offset_s in np.arange(-search_seconds, search_seconds, step_seconds):
        t = DOCUMENTED_COLLISION_TIME + timedelta(seconds=float(offset_s))
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
        e1, p1, v1 = iridium.sgp4(jd, fr)
        e2, p2, v2 = cosmos.sgp4(jd, fr)
        if e1 != 0 or e2 != 0:
            continue
        d = np.linalg.norm(np.array(p1) - np.array(p2))
        if best is None or d < best["d"]:
            best = dict(d=d, offset=offset_s, p1=np.array(p1), p2=np.array(p2),
                        v1=np.array(v1), v2=np.array(v2), t=t)
    if best is None:
        raise RuntimeError("No valid SGP4 samples found for historical validation window.")
    miss_km = best["d"]
    miss_m = miss_km * 1000
    rel_v = np.linalg.norm(best["v1"] - best["v2"])
    return {
        "best": best,
        "miss_km": miss_km,
        "miss_m": miss_m,
        "rel_v": rel_v,
    }


def main():
    iridium = Satrec.twoline2rv(IRIDIUM_L1, IRIDIUM_L2)
    cosmos = Satrec.twoline2rv(COSMOS_L1, COSMOS_L2)
    metrics = compute_historical_metrics()
    best = metrics["best"]
    miss_m = metrics["miss_m"]
    rel_v = metrics["rel_v"]

    print("=" * 70)
    print("HISTORICAL VALIDATION: Iridium 33 / Cosmos 2251, 10 Feb 2009")
    print("=" * 70)
    print(f"Closest approach found at documented time {best['offset']:+.1f}s")
    print(f"  OrbitSense computed miss distance : {miss_m:.1f} m")
    print(f"  Independently verified (AGI STK)   : {DOCUMENTED_STK_MISS_M:.1f} m"
          f"   (agreement: {abs(miss_m - DOCUMENTED_STK_MISS_M):.1f} m)")
    print(f"  OrbitSense computed relative velocity : {rel_v:.3f} km/s")
    print(f"  Documented relative velocity           : ~{DOCUMENTED_REL_VELOCITY_KMS} km/s")
    print()

    print("Probability of Collision (Pc), combined hard-body radius = 10 m:")
    print("(sigma = assumed combined position uncertainty; SOCRATES predictions for")
    print(" this exact event ranged 117 m-1.8 km across the week before, per CelesTrak)")
    for sigma in [300, 500, 700, 1000, 1500, 1800]:
        pc = probability_of_collision(miss_m, sigma, 10.0)
        print(f"  sigma={sigma:5d} m  ->  Pc = {pc:.2e}")

    pc_typical = probability_of_collision(20000, 1000, 10.0)
    print(f"\nFor comparison, a routine 20 km conjunction -> Pc = {pc_typical:.2e}")
    print("=" * 70)

    # ---- visualize the encounter ----
    fig = go.Figure()
    rs = np.random.default_rng(1)
    n_stars = 500
    sr = rs.uniform(30000, 45000, n_stars)
    st = rs.uniform(0, 2 * np.pi, n_stars)
    sp = rs.uniform(0, np.pi, n_stars)
    fig.add_trace(go.Scatter3d(
        x=sr * np.cos(st) * np.sin(sp), y=sr * np.sin(st) * np.sin(sp), z=sr * np.cos(sp),
        mode="markers", marker=dict(size=1, color="white", opacity=0.5),
        hoverinfo="skip", showlegend=False))

    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    fig.add_trace(go.Surface(
        x=R_EARTH * np.cos(u) * np.sin(v), y=R_EARTH * np.sin(u) * np.sin(v), z=R_EARTH * np.cos(v),
        colorscale=[[0, "#1b3a6b"], [1, "#1b3a6b"]], showscale=False, opacity=0.95, hoverinfo="skip"))

    for sat, label, color in [(iridium, "Iridium 33", "#3fa9ff"), (cosmos, "Cosmos 2251", "#ff9d3f")]:
        pts = []
        for m in range(-50, 51):
            t = DOCUMENTED_COLLISION_TIME + timedelta(minutes=m)
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            _, p, _ = sat.sgp4(jd, fr)
            pts.append(p)
        pts = np.array(pts)
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                    mode="lines", line=dict(color=color, width=4),
                                    name=label))

    fig.add_trace(go.Scatter3d(
        x=[best["p1"][0]], y=[best["p1"][1]], z=[best["p1"][2]],
        mode="markers", marker=dict(size=9, color="#ff2b2b", symbol="diamond"),
        text=[f"Collision point<br>Miss: {miss_m:.1f} m<br>Rel. velocity: {rel_v:.2f} km/s<br>"
              f"10 Feb 2009, 16:56 UTC"],
        hoverinfo="text", name="Collision point (actual)"))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   bgcolor="black", aspectmode="data"),
        paper_bgcolor="black", font=dict(color="#c9e8ff"),
        title=dict(text="OrbitSense Historical Validation — Iridium 33 / Cosmos 2251 (10 Feb 2009)",
                   font=dict(size=16, color="white")),
        legend=dict(font=dict(color="white")))
    fig.write_html("historical_validation.html")
    print("\nSaved historical_validation.html — open it in your browser.")


if __name__ == "__main__":
    main()
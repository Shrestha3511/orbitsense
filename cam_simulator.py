import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from scipy import integrate
from scipy.special import ive

MU = 398600.4418  # km^3/s^2

def probability_of_collision(miss_distance_m, sigma_m, hard_body_radius_m):
    d, sig, R = miss_distance_m, sigma_m, hard_body_radius_m
    def integrand(r):
        z = r * d / sig ** 2
        exponent = -(r ** 2 + d ** 2) / (2 * sig ** 2) + z
        return (r / sig ** 2) * np.exp(exponent) * ive(0, z)
    pc, _ = integrate.quad(integrand, 0, R)
    return pc

def rsw_frame(r, v):
    """Local orbital frame at a state vector: Radial, along-track (S), cross-track (W)."""
    r, v = np.array(r), np.array(v)
    R_hat = r / np.linalg.norm(r)
    W_hat = np.cross(r, v); W_hat /= np.linalg.norm(W_hat)
    S_hat = np.cross(W_hat, R_hat)
    return R_hat, S_hat, W_hat

def cw_delta_r(n, t, dv_rsw):
    """Clohessy-Wiltshire: position change at time t from a velocity impulse dv_rsw at t=0."""
    nt = n * t
    Phi_rv = np.array([
        [(1 / n) * np.sin(nt),      (2 / n) * (1 - np.cos(nt)),       0],
        [(2 / n) * (np.cos(nt) - 1), (1 / n) * (4 * np.sin(nt) - 3 * n * t), 0],
        [0,                          0,                                (1 / n) * np.sin(nt)],
    ])
    return Phi_rv @ dv_rsw

IRIDIUM_L1 = "1 24946U 97051C   09040.78448243 +.00000153 +00000-0 +47668-4 0 04775"
IRIDIUM_L2 = "2 24946 086.3994 121.7028 0002288 085.1644 274.9812 14.34219863597336"
COSMOS_L1  = "1 22675U 93036A   09040.49834364 -.00000001 +00000-0 +95251-5 0 07411"
COSMOS_L2  = "2 22675 074.0355 019.4646 0016027 098.7014 261.5952 14.31135643817415"

iridium = Satrec.twoline2rv(IRIDIUM_L1, IRIDIUM_L2)
cosmos = Satrec.twoline2rv(COSMOS_L1, COSMOS_L2)
collision_time = datetime(2009, 2, 10, 16, 55, 59, 800000, tzinfo=timezone.utc)

# validated baseline relative position at the true closest approach (matches STK: 698 m)
# *** FIX: include fractional seconds (collision_time.microsecond) — dropping them was the bug ***
jd0, fr0 = jday(collision_time.year, collision_time.month, collision_time.day,
                 collision_time.hour, collision_time.minute,
                 collision_time.second + collision_time.microsecond / 1e6)
_, r_i, v_i = iridium.sgp4(jd0, fr0)
_, r_c, v_c = cosmos.sgp4(jd0, fr0)
r_rel_true = np.array(r_i) - np.array(r_c)

def simulate_burn(dv_mm_s, direction="along", lead_minutes=120):
    burn_time = collision_time - timedelta(minutes=lead_minutes)
    jd_b, fr_b = jday(burn_time.year, burn_time.month, burn_time.day,
                       burn_time.hour, burn_time.minute,
                       burn_time.second + burn_time.microsecond / 1e6)
    _, r0_i, v0_i = iridium.sgp4(jd_b, fr_b)
    r0_i, v0_i = np.array(r0_i), np.array(v0_i)
    R_hat, S_hat, W_hat = rsw_frame(r0_i, v0_i)
    n = np.sqrt(MU / np.linalg.norm(r0_i) ** 3)

    dv_km_s = dv_mm_s / 1e6
    dv_rsw = {"radial": [dv_km_s, 0, 0], "along": [0, dv_km_s, 0], "cross": [0, 0, dv_km_s]}[direction]
    delta_r_rsw = cw_delta_r(n, lead_minutes * 60, np.array(dv_rsw))
    delta_r_eci = delta_r_rsw[0] * R_hat + delta_r_rsw[1] * S_hat + delta_r_rsw[2] * W_hat

    miss_m = np.linalg.norm(r_rel_true + delta_r_eci) * 1000
    pc = probability_of_collision(miss_m, 1000, 10.0)
    return miss_m, pc

if __name__ == "__main__":
    baseline_miss = np.linalg.norm(r_rel_true) * 1000
    print(f"Baseline (no maneuver): {baseline_miss:.1f} m, "
          f"Pc={probability_of_collision(baseline_miss, 1000, 10.0):.3e}")
    print("\nAlong-track burn, 120 min before encounter:")
    for dv in [0, 10, 30, 50, 100, 300, 500, 1000]:
        miss, pc = simulate_burn(dv, "along", 120)
        print(f"  dv={dv:5d} mm/s -> miss={miss:9.1f} m   Pc={pc:.3e}")
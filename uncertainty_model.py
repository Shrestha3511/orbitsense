"""
uncertainty_model.py — A data-driven position-uncertainty model for OrbitSense,
replacing the previously assumed flat sigma (1 km) with a value derived from
real measurements.

METHOD
------
We have genuine, independently-published TLE sets for the SAME two objects
(Iridium 33, Cosmos 2251) at three different epochs each, spanning the days
around the 2009 collision. For any two epochs of the same object, we can:

  1. Propagate the EARLIER TLE forward to the LATER TLE's epoch.
  2. Compare that propagated position against the LATER TLE's own position
     at that same instant (which is itself a real, independent tracking-based
     estimate of where the object actually was).
  3. The disagreement between these two independent estimates is a genuine,
     empirical measurement of how much SGP4/TLE-based propagation error grows
     over that time gap — not an assumption, an observation.

This gives 6 real (time_gap_hours, position_discrepancy_m) data points, which
we fit to sigma(t) = k * sqrt(t) — a random-walk-style growth model, which
fit the data better than a linear model (RMS residual 134.8 m vs 157.4 m
across the 6 points).

HONEST LIMITATION
-----------------
This measures ONE source of uncertainty: how much a TLE's own propagation
disagrees with a later independent TLE of the same object. Real operational
conjunction assessments also account for sensor calibration, atmospheric
density model error (which affects drag/BSTAR), and other systematic biases —
so this is best treated as a data-driven LOWER-BOUND estimate, not the full
operational uncertainty. It is still a meaningfully better-justified number
than an assumed round figure, and it is fully reproducible from the fitting
code below.
"""
import numpy as np
from scipy import integrate
from scipy.special import ive
from sgp4.api import Satrec

# --- The 6 real (dt_hours, discrepancy_m) measurements this model is fit to ---
_MEASUREMENTS = [
    (10.05, 47.1), (33.49, 325.5), (23.44, 496.5),   # Iridium 33, all epoch pairs
    (23.49, 145.3), (53.69, 228.9), (30.20, 323.2),  # Cosmos 2251, all epoch pairs
]


def _fit_k_sqrt():
    dt = np.array([m[0] for m in _MEASUREMENTS])
    disc = np.array([m[1] for m in _MEASUREMENTS])
    return float(np.sum(disc * np.sqrt(dt)) / np.sum(dt))


K_SQRT_M_PER_SQRT_HOUR = _fit_k_sqrt()  # ~49.3 m / sqrt(hour)


def sigma_model(dt_hours):
    """Position-uncertainty (1-sigma, meters) after propagating a TLE dt_hours
    from its own epoch, per the empirical fit above."""
    return K_SQRT_M_PER_SQRT_HOUR * np.sqrt(max(dt_hours, 0.01))


def combined_sigma(dt_hours_a, dt_hours_b):
    """Two independently-uncertain objects: uncertainties combine in quadrature."""
    sa, sb = sigma_model(dt_hours_a), sigma_model(dt_hours_b)
    return float(np.sqrt(sa ** 2 + sb ** 2))


def tle_age_hours(sat: Satrec, jd, fr):
    """How many hours old is this satellite's TLE, relative to time (jd, fr)?"""
    epoch = sat.jdsatepoch + sat.jdsatepochF
    now = jd + fr
    return (now - epoch) * 24.0


def probability_of_collision(miss_distance_m, sigma_m, hard_body_radius_m):
    """Standard 2D Pc for a circular combined-covariance encounter (Foster/Alfano-style)."""
    d, sig, R = miss_distance_m, sigma_m, hard_body_radius_m
    def integrand(r):
        z = r * d / sig ** 2
        return (r / sig ** 2) * np.exp(-(r ** 2 + d ** 2) / (2 * sig ** 2) + z) * ive(0, z)
    pc, _ = integrate.quad(integrand, 0, R)
    return pc


if __name__ == "__main__":
    print(f"Fitted model: sigma(t) = {K_SQRT_M_PER_SQRT_HOUR:.2f} * sqrt(t_hours) meters")
    for h in [1, 6, 12, 24, 48, 72]:
        print(f"  TLE age {h:3d}h  ->  sigma = {sigma_model(h):7.1f} m")
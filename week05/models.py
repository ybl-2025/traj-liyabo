"""OVM / FVDM / IDM, SI units. See docs/model_equations.md for conventions."""
from dataclasses import dataclass
import math
import numpy as np

MODELS = ("OVM", "FVDM", "IDM")
DEFAULTS = {
    "OVM": dict(v0=30.0, kappa=0.6, h=25.0, ell=10.0),
    "FVDM": dict(v0=30.0, kappa=0.6, h=25.0, ell=10.0, lam=0.6),
    "IDM": dict(v0=30.0, a=1.2, b=2.0, T=1.5, s0=2.0, delta=4.0),
}


def optimal_velocity(s, p):
    """Dimensionalized, normalized Bando tanh curve: V(0)=0, V(inf)=v0."""
    z = math.tanh(p["h"] / p["ell"])
    return p["v0"] * (math.tanh((s - p["h"]) / p["ell"]) + z) / (1 + z)


def acceleration(model, s, v, vl, p):
    if s <= 0:
        raise ValueError("Nonpositive net gap; collision must be handled by integrator")
    if model in ("OVM", "FVDM"):
        result = p["kappa"] * (optimal_velocity(s, p) - v)
        return result + (p["lam"] * (vl - v) if model == "FVDM" else 0.0)
    if model == "IDM":
        desired = p["s0"] + max(0.0, v * p["T"] + v * (v - vl) / (2 * math.sqrt(p["a"] * p["b"])))
        return p["a"] * (1 - (v / p["v0"]) ** p["delta"] - (desired / s) ** 2)
    raise ValueError(model)


@dataclass
class Simulation:
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    gap: np.ndarray
    status: str
    failure_time_s: float | None
    min_gap_m: float
    max_abs_accel_m_s2: float
    stop_corrections: int


def simulate(model, p, t, leader_x, leader_v, leader_length, x0, v0, dt=0.05):
    """Ballistic substeps, piecewise-linear leader position/speed replay.

    No safety-gap or acceleration clipping. Negative speeds use exact within-step
    stopping; collisions terminate the rollout and uncomputed observations are NaN.
    Continuous within-substep gap minimum is checked with a linear leader position.
    """
    t = np.asarray(t, float)
    lx, lv = np.asarray(leader_x, float), np.asarray(leader_v, float)
    ll = np.broadcast_to(np.asarray(leader_length, float), t.shape)
    if dt <= 0 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Require increasing times and dt > 0")
    if not all(np.isfinite(a).all() for a in (t, lx, lv, ll)):
        raise ValueError("Nonfinite leader inputs")
    x, v, acc, gap = (np.full(len(t), np.nan) for _ in range(4))
    x[0], v[0], gap[0] = x0, v0, lx[0] - ll[0] - x0
    if not (np.isfinite(x0) and np.isfinite(v0) and v0 >= 0):
        raise ValueError("Invalid follower initial condition")
    min_gap, max_a, stops = gap[0], 0.0, 0
    status, failtime = "success", None
    if min_gap <= 0:
        return Simulation(t, x, v, acc, gap, "collision", float(t[0]), float(min_gap), 0., 0)
    xx, vv = float(x0), float(v0)
    for i in range(len(t) - 1):
        span = t[i + 1] - t[i]
        n = max(1, int(math.ceil(span / dt - 1e-10)))
        hh = span / n
        for j in range(n):
            f0, f1 = j / n, (j + 1) / n
            ll0 = ll[i] + (ll[i + 1] - ll[i]) * f0
            ll1 = ll[i] + (ll[i + 1] - ll[i]) * f1
            rear0 = lx[i] + (lx[i + 1] - lx[i]) * f0 - ll0
            rear1 = lx[i] + (lx[i + 1] - lx[i]) * f1 - ll1
            vl = lv[i] + (lv[i + 1] - lv[i]) * f0
            aa = acceleration(model, rear0 - xx, vv, vl, p)
            if not math.isfinite(aa) or abs(aa) > 1e8:
                status, failtime = "numerical_failure", float(t[i] + j * hh)
                break
            max_a = max(max_a, abs(aa))
            moving = min(hh, -vv / aa) if aa < 0 and vv + aa * hh < 0 else hh
            if moving < hh:
                stops += 1
            newx = xx + vv * moving + 0.5 * aa * moving * moving
            newv = max(0., vv + aa * hh)
            # On [0,moving], gap is quadratic; on [moving,hh] it is linear.
            rear_speed = (rear1 - rear0) / hh
            candidates = [0., moving, hh]
            if aa != 0:
                extremum = (rear_speed - vv) / aa
                if 0 < extremum < moving:
                    candidates.append(extremum)
            def subgap(q):
                z = min(q, moving)
                return rear0 + rear_speed * q - xx - vv * z - .5 * aa * z * z
            local_min = min(subgap(q) for q in candidates)
            min_gap = min(min_gap, local_min)
            if local_min <= 0:
                # Earliest crossing on monotonic pieces, then bisection.
                knots = sorted(set(candidates))
                lo, hi = next((l, r) for l, r in zip(knots[:-1], knots[1:]) if subgap(r) <= 0)
                for _ in range(35):
                    mid = .5 * (lo + hi)
                    if subgap(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                status, failtime = "collision", float(t[i] + j * hh + hi)
                # Physical trajectory ends at first contact, not at substep endpoint.
                min_gap = 0.0
                break
            xx, vv = newx, newv
        if status != "success":
            break
        x[i + 1], v[i + 1] = xx, vv
        gap[i + 1] = lx[i + 1] - ll[i + 1] - xx
        acc[i] = acceleration(model, gap[i], v[i], lv[i], p)
    if status == "success":
        acc[-1] = acceleration(model, gap[-1], v[-1], lv[-1], p)
        max_a = max(max_a, abs(acc[-1]))
    return Simulation(t, x, v, acc, gap, status, failtime, float(min_gap), float(max_a), stops)

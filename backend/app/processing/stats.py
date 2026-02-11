import time
import math
import numpy as np


# =========================================================
# Geometry helpers
# =========================================================

def calculate_angle(a, b, c):
    ax, ay = a["x"], a["y"]
    bx, by = b["x"], b["y"]
    cx, cy = c["x"], c["y"]

    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def compute_com(keypoints):
    indices = [11, 12, 23, 24]
    pts = []

    for idx in indices:
        if idx < len(keypoints) and keypoints[idx] is not None:
            pts.append((keypoints[idx]["x"], keypoints[idx]["y"]))

    if not pts:
        return None

    xs, ys = zip(*pts)
    return np.array([float(sum(xs) / len(xs)), float(sum(ys) / len(ys))])


# =========================================================
# Kalman Filter for COM
# =========================================================

class KalmanCOM:
    def __init__(self):
        self.x = None
        self.P = 1.0
        self.Q = 0.01
        self.R = 0.1

    def update(self, z):
        if z is None:
            return self.x

        z = np.array(z, dtype=float)

        if self.x is None:
            self.x = z
            return self.x

        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

        return self.x
    
    def reset(self):
        """Reset filter state (call when person leaves frame)"""
        self.x = None
        self.P = 1.0


# =========================================================
# Support persistence (NEW)
# =========================================================

class SupportTracker:
    """
    Stabilizes base-of-support over time.
    Prevents BOS flicker due to brief occlusion or jitter.
    """

    def __init__(self, drop_time=0.35):
        self.current_support = None
        self.last_seen_time = None
        self.drop_time = drop_time

    def update(self, support_type, bos_polygon):
        now = time.time()

        if support_type and bos_polygon:
            self.current_support = {
                "support": support_type,
                "polygon": bos_polygon
            }
            self.last_seen_time = now
            return self.current_support

        if (
            self.current_support and
            self.last_seen_time and
            now - self.last_seen_time < self.drop_time
        ):
            return self.current_support

        self.current_support = None
        self.last_seen_time = None
        return None


# =========================================================
# Stability reasoning
# =========================================================

class StabilityTracker:
    def __init__(
        self,
        unstable_time=0.25,
        margin_eps=0.015,
        hysteresis=0.01
    ):
        self.outside_since = None
        self.state = "stable"

        self.unstable_time = unstable_time
        self.margin_eps = margin_eps
        self.hysteresis = hysteresis

    def update(self, com_x, bos_polygon):
        if com_x is None or not bos_polygon:
            self.outside_since = None
            self.state = "stable"
            return self.state, None

        xs = [p[0] for p in bos_polygon]
        min_x = min(xs)
        max_x = max(xs)

        if min_x <= com_x <= max_x:
            margin = min(com_x - min_x, max_x - com_x)
        else:
            margin = -min(abs(com_x - min_x), abs(com_x - max_x))

        now = time.time()

        if margin >= self.margin_eps + self.hysteresis:
            self.outside_since = None
            self.state = "stable"

        elif margin >= -self.margin_eps:
            self.outside_since = None
            self.state = "marginal"

        else:
            if self.outside_since is None:
                self.outside_since = now
                self.state = "marginal"
            elif now - self.outside_since >= self.unstable_time:
                self.state = "unstable"
            else:
                self.state = "marginal"

        return self.state, float(margin)


# =========================================================
# Phase segmentation
# =========================================================

class PhaseTracker:
    def __init__(self, recovery_time=0.4):
        self.current_phase = "double_support_stable"
        self.last_unstable_time = None
        self.recovery_time = recovery_time

    def update(self, support_type, stability_state, com_speed):
        now = time.time()

        if stability_state == "unstable":
            self.current_phase = "unstable"
            self.last_unstable_time = now
            return self.current_phase

        if self.last_unstable_time is not None:
            if now - self.last_unstable_time < self.recovery_time:
                self.current_phase = "recovery"
                return self.current_phase
            self.last_unstable_time = None

        if stability_state == "marginal":
            self.current_phase = "transition"
            return self.current_phase

        if com_speed and com_speed > 0.25:
            self.current_phase = "transition"
            return self.current_phase

        if support_type == "double_foot":
            self.current_phase = "double_support_stable"
        elif support_type == "single_foot":
            self.current_phase = "single_support_stable"
        else:
            self.current_phase = "transition"

        return self.current_phase

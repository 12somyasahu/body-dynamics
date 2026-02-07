import time
import math
import numpy as np


# =========================================================
# Geometry helpers lfgggggggggg
# =========================================================

def calculate_angle(a, b, c):
    """
    Calculate angle at point b (in degrees) given points a, b, c.
    Points are dicts or tuples with x, y.
    """
    ax, ay = a["x"], a["y"]
    bx, by = b["x"], b["y"]
    cx, cy = c["x"], c["y"]

    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def compute_com(keypoints):
    """
    Approximate Center of Mass using shoulders and hips.
    keypoints: list of dicts or None
    """
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
    """
    Simple Kalman filter for COM smoothing.
    """
    def __init__(self):
        self.x = None
        self.P = 1.0
        self.Q = 0.01   # process noise
        self.R = 0.1    # measurement noise

    def update(self, z):
        if z is None:
            return self.x

        z = np.array(z, dtype=float)

        if self.x is None:
            self.x = z
            return self.x

        # Prediction
        self.P += self.Q

        # Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

        return self.x


# =========================================================
# Stability reasoning (THIS is the upgrade)
# =========================================================

class StabilityTracker:
    """
    Tracks stability over time using COM vs BOS margin.

    States:
      - stable
      - marginal
      - unstable
    """

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
        """
        com_x: float
        bos_polygon: list of (x, y) tuples
        returns: (state, margin)
        """

        if com_x is None or not bos_polygon:
            self.outside_since = None
            self.state = "stable"
            return self.state, None

        xs = [p[0] for p in bos_polygon]
        min_x = min(xs)
        max_x = max(xs)

        # Margin: positive = inside, negative = outside
        if min_x <= com_x <= max_x:
            margin = min(com_x - min_x, max_x - com_x)
        else:
            margin = -min(abs(com_x - min_x), abs(com_x - max_x))

        now = time.time()

        # ==============================
        # State machine with hysteresis
        # ==============================

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

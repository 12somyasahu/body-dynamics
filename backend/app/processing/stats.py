import math
import numpy as np

# ----------------------------
# ANGLE UTILITY
# ----------------------------

def calculate_angle(a, b, c):
    """
    Calculate angle ABC (in degrees)
    a, b, c are [x, y]
    """

    try:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag_ba == 0 or mag_bc == 0:
            return None

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle = math.degrees(math.acos(cos_angle))
        return angle

    except Exception:
        return None


# ----------------------------
# CENTER OF MASS (COM)
# ----------------------------

def compute_com(keypoints):
    """
    Compute Center of Mass (COM) using torso landmarks.

    Uses:
    - Left shoulder  (11)
    - Right shoulder (12)
    - Left hip       (23)
    - Right hip      (24)

    Returns: (x, y) normalized or None
    """

    REQUIRED = [11, 12, 23, 24]

    try:
        points = []
        for idx in REQUIRED:
            if idx >= len(keypoints):
                return None
            if keypoints[idx] is None:
                return None
            points.append(keypoints[idx])

        x = sum(p[0] for p in points) / 4
        y = sum(p[1] for p in points) / 4

        return (x, y)

    except Exception:
        return None
  


class KalmanCOM:
    """
    2D Kalman Filter for Center of Mass tracking.
    State: [x, y, vx, vy]
    """

    def __init__(self):
        # State vector
        self.x = np.zeros((4, 1))

        # State covariance
        self.P = np.eye(4) * 1.0

        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise
        self.R = np.eye(2) * 0.01

        # Process noise
        self.Q = np.eye(4) * 0.001

        self.initialized = False

    def update(self, measurement):
        """
        measurement: (x, y) tuple
        returns: (x, y) filtered
        stats ka satta
        """

        z = np.array([[measurement[0]], [measurement[1]]])

        # Initialize state
        if not self.initialized:
            self.x[0, 0] = measurement[0]
            self.x[1, 0] = measurement[1]
            self.initialized = True
            return measurement

        # ----------------
        # Predict
        # ----------------
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ----------------
        # Update
        # ----------------
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return (self.x[0, 0], self.x[1, 0])


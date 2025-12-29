import math


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

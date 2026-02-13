import math
import numpy as np


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

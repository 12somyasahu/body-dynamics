import time


class StabilityTracker:
    def __init__(self, unstable_time=0.25, margin_eps=0.015, hysteresis=0.01):
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

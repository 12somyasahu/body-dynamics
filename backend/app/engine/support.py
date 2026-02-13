import time


class SupportTracker:
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

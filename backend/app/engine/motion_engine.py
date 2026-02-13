import time
from .support import SupportTracker
from .stability import StabilityTracker
from .phase import PhaseTracker


class MotionEngine:

    def __init__(self):
        self.support_tracker = SupportTracker()
        self.stability_tracker = StabilityTracker()
        self.phase_tracker = PhaseTracker()

        self.com_history = []

    def update(
        self,
        keypoints,
        filtered_com,
        raw_support,
        raw_bos,
        angle,
        config
    ):

        # Support persistence
        support = self.support_tracker.update(raw_support, raw_bos)
        support_type = support["support"] if support else None
        bos_polygon = support["polygon"] if support else None

        # Stability
        stab_state, margin = None, None
        if filtered_com is not None and bos_polygon:
            stab_state, margin = self.stability_tracker.update(
                float(filtered_com[0]),
                bos_polygon
            )

        # Phase
        phase = self.phase_tracker.update(
            support_type,
            stab_state,
            0
        )

        # COM trail
        if filtered_com is not None:
            self.com_history.append({
                "x": round(float(filtered_com[0]), 3),
                "y": round(float(filtered_com[1]), 3),
                "t": round(time.time(), 3)
            })
            if len(self.com_history) > config.max_trail:
                self.com_history.pop(0)

        return {
            "support": support_type,
            "stability": {
                "state": stab_state,
                "margin": round(margin, 3) if margin else None
            } if stab_state else None,
            "phase": phase,
            "com_trail": self.com_history
        }

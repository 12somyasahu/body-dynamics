import time


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

import numpy as np


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
        self.x = None
        self.P = 1.0

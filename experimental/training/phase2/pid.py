class EdgePID:
    def __init__(
        self, kp: float = 0.1, ki: float = 0.0, kd: float = 0.02, target: float = 0.5
    ):
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, target
        self.int_err = 0.0
        self.last_err = 0.0

    def update(self, complexity: float) -> float:
        err = self.target - complexity
        d_err = err - self.last_err
        self.int_err += err
        self.last_err = err
        return self.kp * err + self.ki * self.int_err + self.kd * d_err

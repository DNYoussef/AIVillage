class EdgePID:
    def __init__(self, kp=0.1, ki=0.0, kd=0.02, target=0.5) -> None:
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, kd
        self.int_err = self.last_err = 0.0

    def update(self, complexity):
        err = self.target - complexity
        d_err = err - self.last_err
        self.int_err += err
        self.last_err = err
        return self.kp * err + self.ki * self.int_err + self.kd * d_err

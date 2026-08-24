class PI_controller:
    def __init__(self, Kp, Ki, integral_limit=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, delta_t=0.01):
        # -------------------------------------------------
        # Integral term
        # -------------------------------------------------
        self.integral += error * delta_t

        # Reset integral when error changes sign
        if error * self.prev_error < 0:
            self.integral = 0.0

        # Clamp integral term
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral)
        )

        # -------------------------------------------------
        # PI controller
        # -------------------------------------------------
        delta_p = self.Kp * error + self.Ki * self.integral

        # Store current error for next iteration
        self.prev_error = error

        return delta_p
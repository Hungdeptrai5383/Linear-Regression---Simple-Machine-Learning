class LinearRegression:
    def __init__(self, m = 0, b = 0):
        self.m = m
        self.b = b

    def predict(self, x):
        return self.m * x + self.b
from model import LinearRegression
from calculate_error import MSE

def train(model, x_data, y_data, learning_rate, epochs):
    n = len(x_data)
    for _ in range(epochs):
        y_predict = []
        for x in x_data:
            y_predict.append(model.predict(x))
        gradient_m = 0
        gradient_b = 0

        for i in range(n):
            error = y_data[i] - y_predict[i]
            gradient_m += x_data[i] * error
            gradient_b += error
        gradient_m *= (-2/n)
        gradient_b *= (-2/n)

        model.m -= learning_rate * gradient_m
        model.b -= learning_rate * gradient_b
    return model
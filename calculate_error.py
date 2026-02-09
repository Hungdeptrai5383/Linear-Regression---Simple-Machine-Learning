# Mean Square Error
def MSE(y_true, y_predict):
    n = len(y_true)
    total_error = 0

    for i in range (n):
        diff = y_true[i] - y_predict[i]
        total_error += diff ** 2

    return total_error / n
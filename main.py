from model import LinearRegression
from train import train

x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 6, 8, 10]  

model = LinearRegression()

trained_model = train(
    model,
    x_data,
    y_data,
    learning_rate=0.01,
    epochs=1000
)

print("Trained m:", trained_model.m)
print("Trained b:", trained_model.b)

test_x = 7
prediction = trained_model.predict(test_x)

print("Prediction for x=7:", prediction)

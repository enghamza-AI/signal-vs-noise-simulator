import random


def true_function(x):
    return 2 * x + 1


def generate_data(num_points, noise_level):
    x_values = []
    y_values = []

    for _ in range(num_points):
        x = random.uniform(-10, 10)
        true_y = true_function(x)

        noise = random.uniform(-noise_level, noise_level)
        y_noisy = true_y + noise

        x_values.append(x)
        y_values.append(y_noisy)

    return x_values, y_values


def train_linear_regression(x, y, learning_rate=0.001, epochs=1000):
    m = 0
    b = 0
    n = len(x)

    for _ in range(epochs):
        m_grad = 0
        b_grad = 0

        for i in range(n):
            pred = m * x[i] + b
            error = pred - y[i]

            m_grad += error * x[i]
            b_grad += error

        m_grad = (2 / n) * m_grad
        b_grad = (2 / n) * b_grad

        m -= learning_rate * m_grad
        b -= learning_rate * b_grad

    return m, b


def predict(x, m, b):
    return [m * xi + b for xi in x]


def mean_squared_error(y_true, y_pred):
    total = 0
    n = len(y_true)

    for i in range(n):
        diff = y_true[i] - y_pred[i]
        total += diff * diff

    return total / n


def print_sample(x, y_true, y_pred, count=10):
    print("\nSample predictions:\n")
    for i in range(count):
        print(f"x={x[i]:.2f} | true={y_true[i]:.2f} | pred={y_pred[i]:.2f}")




noise_level = 10

x, y = generate_data(100, noise_level)

m, b = train_linear_regression(x, y)

preds = predict(x, m, b)

true_y = [true_function(xi) for xi in x]

mse = mean_squared_error(true_y, preds)

print("Learned slope:", m)
print("Learned intercept:", b)
print("MSE:", mse)

print_sample(x, true_y, preds)

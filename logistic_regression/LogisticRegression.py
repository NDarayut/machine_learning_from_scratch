import numpy as np

class LogisticRegression:

    def __init__(self, learning_rates = 0.001, iterations = 100):
        self.learning_rates = learning_rates
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, Z):
        return 1 / (1+np.exp(-Z))

    def fit(self, X, y):

        """
        args:
            -- X: Data point which contains random values of number 
            -- y: Class label of X features (1 or 0)
        """
        number_of_samples, number_of_features = X.shape

        # Initialize weights centered around 0
        self.weights = np.random.uniform(-1, 1, number_of_features)
        self.bias = 0

        for i in range(self.iterations):
            cost = 0

            Z = np.dot(self.weights, X.T) + self.bias
            y_hat = self.sigmoid(Z)

            # Gradeint descent
            d_weights = (1/number_of_samples) * np.dot(X.T, (y_hat - y))
            d_bias = (1/number_of_samples) * np.sum(y_hat - y)

            new_weights = self.weights - (self.learning_rates * d_weights)
            new_bias = self.bias - (self.learning_rates * d_bias)

            self.weights = new_weights
            self.bias = new_bias

            # Logistic cost function
            cost = -(1/number_of_samples) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

            print(f"Iterations: {i + 1} Cost: {cost}")

    def predict(self, X, threshold):
        """
        args:
            -- X: The input data point
            -- threshold: The boundary that dictate if the activated input True or False
        """
        Z = np.dot(self.weights, X.T) + self.bias
        activation = sigmoid(Z)
        y_pred = [1 if i > threshold else 0 for i in activation]

        return y_pred

if __name__ == "__main__":

    import argparse
    import matplotlib.pyplot as plt

    # Allow user to change parameters in terminal without interacting with the code
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model.")
    parser.add_argument("--learning_rates", type=float, default=0.01, help="Learning rate for gradient descent")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for training")

    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(6)

    number_of_samples = 200
    number_of_features = 2

    ratio_of_class_0 = 0.5

    # Generate features (X) for class 0 and class 1
    X_class_0 = np.random.randn(int(number_of_samples * ratio_of_class_0), number_of_features) + np.array([-2, -2])
    X_class_1 = np.random.randn(int(number_of_samples * 1 - ratio_of_class_0), number_of_features) + np.array([2, 2])

    # Combine the features and create labels
    X = np.vstack((X_class_0, X_class_1))
    y = np.array([0] * len(X_class_0) + [1] * len(X_class_1))  # Labels: 0 for class 0, 1 for class 1

    # Shuffle the data
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    model = LogisticRegression(learning_rates = args.learning_rates, iterations = args.iterations)
    model.fit(X, y)

    # Visualize the dataset
    plt.figure(figsize=(8, 6))

    # Plot points for class 0
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')
    # Plot points for class 1
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')

    # Generate a grid of points to evaluate the model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities for each point in the grid
    Z = np.dot(grid_points, model.weights) + model.bias
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary (where the probability = 0.5)
    plt.contourf(xx, yy, model.sigmoid(Z), levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    plt.contour(xx, yy, model.sigmoid(Z), levels=[0.5], colors='black', linewidths=1, linestyles='--')

    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()




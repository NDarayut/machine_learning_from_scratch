import numpy as np

class LogisticRegression:

    def __init__(self, learning_rates, iterations):
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

    # Allow user to change parameters in terminal without interacting with the code
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model.")
    # Initialize default learning rates and iterations
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
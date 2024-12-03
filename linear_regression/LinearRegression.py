import numpy as np

class LinearRegression:

    # Constructor to initializes the default learning rates and iterations of the model
    def __init__(self, learning_rates = 0.001, iterations = 100):
        self.learning_rates = learning_rates
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    # Fit function that takes independent variable (X) and dependent variable (y)
    def fit(self, X, y):
        """
            -- number_of_features: The amount of features that made up a data point (e.g., size, width, or height)
            -- number_of_samples: The amount of data within the dataset (e.g., 1000 houses..etc)
        """
        number_of_samples, number_of_features = X.shape

        # Generate random weights between -1 and 1 equal to the amount of the features
        self.weights = np.random.uniform(-1, 1, number_of_features)
        self.bias = 0

        for i in range(self.iterations):
            
            cost = 0

            # np.dot() = sum(Wi * Xi)
            # e.g., X has 1 samples and 2 features which looks like [[2, 1]] 
            # weights would be [-0.12, 1.47] which has a different dimension from X
            # Hence we transpose X so that it would be [[2], [1]] to have the same dimension as weights
            y_hat = np.dot(self.weights, X.T) + self.bias

            # Calculate the cost (error) of the model across all data point
            cost = (1/(2*number_of_samples)) * np.sum((y_hat - y)**2)

            # Derivatves of the cost function with respect to the weights
            # d_weights = (1/number_of_samples) * np.sum((y_hat - y)*X)

            # Vectorized version of the derivative of the cost function w.r.t weights
            d_weights = (1/number_of_samples) * np.dot(X.T, (y_hat - y))

            # Derivatves of the cost function with respect to the bias
            d_bias = (1/number_of_samples) * np.sum((y_hat - y))

            # Updates weights and biases
            new_weights = self.weights - (self.learning_rates * d_weights)
            new_bias = self.bias - (self.learning_rates * d_bias)

            self.weights = new_weights
            self.bias = new_bias

            print (f"Iteration: {i} Cost: {cost}")

    def predict(self, X):
        y_pred = np.dot(self.weights, X.T) + self.bias
        return y_pred

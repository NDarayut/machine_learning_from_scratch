from LinearRegression import LinearRegression
import numpy as np

# Ensure reproducibility
np.random.seed(123)

# Parameters
number_of_samples = 100  # Number of data points (samples)
number_of_features = 10   # Number of features

# Generate random feature values (e.g., normally distributed)
X = np.random.rand(number_of_samples, number_of_features)

# Define the weights and bias for the linear model
weights = np.random.rand(number_of_features)  # Coefficient for the feature
bias = 5 # Intercept

# Generate the target values using a linear relation with some noise
# np.random.randn(number_of_samples): Generate a bunch of random numbers equal to the amount of samples
# 0.2 makes the number less varied
noise = np.random.randn(number_of_samples) * 0.2  

# Generate the true value e.g., y
y = np.dot(weights, X.T) + bias + noise

# Create the model
model = LinearRegression(learning_rates=0.1, iterations=1000) # learning_rates and iterations are hyperparameter
model.fit(X, y)


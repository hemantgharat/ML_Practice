# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a synthetic dataset with multiple features
np.random.seed(0)
X1 = 2 * np.random.rand(100, 1)  # Feature 1
X2 = 3 * np.random.rand(100, 1)  # Feature 2
X = np.concatenate([X1, X2], axis=1)  # Combine features into a single dataset
y = 5 + 2 * X1 + 4 * X2 + np.random.randn(100, 1)  # y = 5 + 2*X1 + 4*X2 + noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the multiple linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics and coefficients
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plotting the results (only possible with 2D data, so we’ll keep it basic)
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_test)), y_pred, color="red", label="Predicted", alpha=0.5)
plt.xlabel("Test Sample Index")
plt.ylabel("y (Dependent Variable)")
plt.legend()
plt.title("Multiple Linear Regression Predictions vs Actual")
plt.show()

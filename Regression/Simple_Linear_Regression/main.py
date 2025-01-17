# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a simple dataset
# Let's assume X is the feature and y is the target variable
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random values for X
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3*X + noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plotting the results
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()

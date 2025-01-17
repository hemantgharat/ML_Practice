# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate a synthetic dataset
np.random.seed(0)
X = 6 * np.random.rand(100, 1) - 3  # Feature
y = 0.5 * X**3 - X**2 + 2 * X + np.random.randn(100, 1)  # Cubic function with noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the features to polynomial features
degree = 3  # You can adjust this for higher or lower polynomial degrees
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the polynomial features
model.fit(X_train_poly, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plotting the results
plt.scatter(X, y, color="blue", label="Data Points")
X_line = np.linspace(-3, 3, 100).reshape(100, 1)
y_line = model.predict(poly_features.transform(X_line))
plt.plot(X_line, y_line, color="red", label=f"Polynomial Regression (Degree {degree})")
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.legend()
plt.title(f"Polynomial Regression (Degree {degree})")
plt.show()

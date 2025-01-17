# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# Assuming you have a CSV file with features like 'attendance', 'homework_score', 'exam_score', and 'at_risk' label
data = pd.read_csv(r'ML_Practice/Regression/Logistic_Regression/student_data.csv')
# Define the features (X) and the target (y)
# X includes columns that can help predict risk (e.g., attendance rate, homework completion, test scores)
# y is the binary label indicating if the student is at risk (1) or not (0)
X = data[['attendance', 'homework_score', 'exam_score']]
y = data['at_risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 5000 students
n_samples = 5000

# Create random data for attendance (0 to 100), homework scores (0 to 100), and exam scores (0 to 100)
attendance = np.random.randint(60, 100, size=n_samples)
homework_score = np.random.randint(50, 100, size=n_samples)
exam_score = np.random.randint(40, 100, size=n_samples)

# Generate a binary 'at_risk' target variable based on some simple rule (e.g., low attendance or low scores)
at_risk = ((attendance < 75) | (homework_score < 60) | (exam_score < 60)).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'attendance': attendance,
    'homework_score': homework_score,
    'exam_score': exam_score,
    'at_risk': at_risk
})

# Save the dataset to a CSV file
data.to_csv('student_data_5000.csv', index=False)

# Show the first few rows of the generated dataset
print(data.head())

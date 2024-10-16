
# Import necessary libraries. Begin by importing all the necessary libraries - pandas for data manipulation, train_test_split for splitting our dataset into training and test sets, and StandardScaler for feature scaling.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset using the `pd.read_csv` function from pandas. The dataset is read directly from a CSV file named 'iris.csv'. This file is assumed to be in the same directory as the script. The dataset is stored in a pandas DataFrame for easier manipulation
iris_df = pd.read_csv(r'ML_Practice/Data_Preprocessing/ML_3/iris.csv')

# Separate features and target. The independent variables (features) and dependent variable (target) are separated into X and y, respectively. It is assumed that the target variable is named 'target' in the dataset.
X = iris_df.drop('target', axis=1)  # Assuming 'target' is the column name for the target variable
y = iris_df['target']

# Split the dataset into an 80-20 training-test set. Use the train_test_split function to split our dataset into training and test sets. We specify a test_size of 0.2, which means 80% of the data will be used for training and 20% will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling. The StandardScaler is applied to standardize the features to have a mean=0 and variance=1. The scaler is fitted on the training set and then used to transform both the training and test sets. This is to prevent information leak from the test set into the training set.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the scaled training and test sets. Finally, the scaled training and test sets are printed to verify the scaling.
print("Scaled Training Set:")
print(X_train)
print("\nScaled Test Set:")
print(X_test)
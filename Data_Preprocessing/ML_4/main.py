# Import necessary libraries: We start by importing the necessary libraries for data preprocessing. This includes pandas for data manipulation, train_test_split from sklearn.model_selection to split our dataset into training and test sets, and StandardScaler from sklearn.preprocessing to apply feature scaling.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset: The Wine Quality Red dataset is loaded into a pandas DataFrame using the pd.read_csv function. Here, we need to specify the correct delimiter, which in this case is a semicolon.
df = pd.read_csv(r'ML_Practice/Data_Preprocessing/ML_4/winequality-red.csv', delimiter=';')

# Split the dataset into a training set and a test set: We separate the target variable 'quality' from the features and then split the dataset into an 80-20 training-test set using the train_test_split function.
X = df.drop('quality', axis=1) 
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the StandardScaler class: The StandardScaler class is used to standardize features by removing the mean and scaling to unit variance.
sc = StandardScaler()

# Fit the StandardScaler on the training set and transform the data: The StandardScaler is fitted to the training set and then used to transform both the training and test datasets.
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# Print the scaled datasets: Finally, we print the scaled training and test datasets to verify the feature scaling process.

print("Scaled training set:\n", X_train) 
print("Scaled test set:\n", X_test)
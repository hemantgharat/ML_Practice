# Import the necessary libraries: We start by importing the necessary libraries for this exercise. Pandas is a library providing high-performance, easy-to-use data structures and data analysis tools. NumPy is a library used for working with arrays. SimpleImputer is a class from the sklearn.impute module that provides basic strategies for imputing missing values.
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset: The dataset is loaded into a pandas DataFrame using the read_csv function. This function is widely used in pandas to read a comma-separated values (csv) file into DataFrame.
df = pd.read_csv(r'ML_Practice/Data_Preprocessing/ML_2/ML_2.1/pima-indians-diabetes.csv')

# Identify missing data: We identify missing data in the DataFrame using the isnull function followed by the sum function. This gives us the number of missing entries in each column. These missing entries are represented as NaN.
missing_data = df.isnull().sum()

# Print the number of missing entries in each column
print("Missing data: \n", missing_data)

# Configure an instance of the SimpleImputer class: We create an instance of the SimpleImputer class. This class is a part of the sklearn.impute module and provides basic strategies for imputing missing values. We configure it to replace missing values (represented as np.nan) with the mean value of the column.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame: We fit the imputer on the DataFrame using the fit method. This method calculates the imputation values (in this case, the mean of each column) that will be used to replace the missing data.
imputer.fit(df)

# Apply the transform to the DataFrame: We apply the transform to the DataFrame using the transform method. This method replaces missing data with the imputation values calculated by the fit method.
df_imputed = imputer.transform(df)

# Print the updated matrix of features: Finally, we print out the updated matrix of features to verify that the missing data has been successfully replaced.

print("Updated matrix of features: \n", df_imputed)
# Import necessary libraries: We first import pandas library which provide high-performance, easy-to-use data structures and data analysis tools.
import pandas as pd

# Load the dataset: We use the read_csv function from pandas to load the Iris dataset from a CSV file. The loaded dataset is a DataFrame object which is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).
dataset = pd.read_csv(r'ML_Practice/Data_Preprocessing/ML_1/iris.csv')

# Create the matrix of features and dependent variable vector: We extract the matrix of features (X) and dependent variable vector (y) from the DataFrame. The iloc indexer is used to select all rows and all columns except the last one for X, and all rows of the last column for y. The .values attribute is used to extract the data as numpy arrays.
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

# Print X and y: Finally, we print out the matrix of features and dependent variable vector to verify their creation.
print(X)
print(y)
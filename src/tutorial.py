import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

print(df.mean())

only_values = df.drop('target', 1)

substracted_means = only_values - only_values.mean()

for i, row in substracted_means.iterrows():
    for j, column in row.iteritems():
        print(column)

covariance_matrix = np.cov(substracted_means)
print(covariance_matrix)

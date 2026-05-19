import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression_from_scratch import LinearRegression
from sklearn.model_selection import train_test_split

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)
print(predicted)
plt.scatter(x_test, y_test, color='red', marker='o', label='Test data')
plt.plot(x_test, predicted, color='blue', label='Predicted')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

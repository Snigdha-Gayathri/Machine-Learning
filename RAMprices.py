import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\snigd\ML\ram_price.csv')


data.dropna(inplace=True)


x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=18)


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(xtrain, ytrain)


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(xtrain, ytrain)


pred_tree = tree.predict(xtest)
pred_lr = linear_reg.predict(xtest)

from sklearn.metrics import r2_score
r_squared = r2_score(ytest, pred_tree)
print('R-squared Score: ', r_squared)


from sklearn.metrics import r2_score
r_squared = r2_score(ytest, pred_lr)
print('R-squared Score: ', r_squared)

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes


dataset=load_diabetes()

df=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)

x=df
y=dataset.target

print(df.head())
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=18)

##from sklearn.linear_model import LinearRegression
##model=LinearRegression()


##from sklearn.tree import DecisionTreeRegressor
##model=DecisionTreeRegressor()

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)


from sklearn.metrics import r2_score
r_squared = r2_score(ytest, ypred)
print('R-squared Score: ', r_squared)

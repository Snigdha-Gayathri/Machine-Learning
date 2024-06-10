import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import fetch_openml
dataset = fetch_openml('titanic', version=1)

df=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)

x=df
y=dataset.target

print(dataset.data.shape)  
print(dataset.target.shape) 
print(dataset.DESCR) 
print(dataset.feature_names) 
print(dataset.target_names)

from skelarn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split=train_test_split(x,y,test_size=0.2,random_state=9)

from sklearn.linear_model import LinearRegression
model=LinearRegressor()


#from sklearn.tree import DecisionTreeRegressor
#model=DecisionTreeRegressor()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

min_vals = df.min().values
max_vals = df.max().values
random_row = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]

print(random_row)

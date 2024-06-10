import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_wine
dataset = load_wine()


df=pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

x=df
y=dataset.target

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

min_vals = df.min().values
max_vals = df.max().values
random_row = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]


#print(random_row)

prediction=model.predict([[12.451070691515431, 3.703303109613401, 3.015387950901062, 16.16965056734422, 95.61097019815261, 3.4312189062024014, 4.144965031323723, 0.1603913026180176, 2.17625948195125, 11.03503667641351, 0.7734526025067161, 2.5607658605912316, 1383.8647461676403]])

print(prediction)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

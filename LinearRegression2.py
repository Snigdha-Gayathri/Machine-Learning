import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data=pd.read_csv(r"C:\Users\snigd\ML\1000_Companies.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])

onehotencoder=OneHotEncoder()
x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.20,random_state=4)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


pred=model.predict(xtest)

print(model.coef_)
print(model.intercept_)

from sklearn.metrics import r2_score
print(r2_score(ytest,ypred)*100)



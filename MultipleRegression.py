import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\snigd\ML\ageTarget.csv')

x=np.array(data.iloc[:,:-1])

y=data.iloc[:,-1].values

#Target variable=Age


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=18)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)


prediction=model.predict([[18,63700]])
print(prediction)



data=pd.read_csv(r'C:\Users\snigd\ML\expTarget.csv')

x=np.array(data.iloc[:,:-1])

y=data.iloc[:,-1].values
#Target variable=Experience

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=18)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

prediction=model.predict([[36730,54]])
print(prediction)


data=pd.read_csv(r'C:\Users\snigd\ML\incomeTarget.csv')

x=np.array(data.iloc[:,:-1])

y=data.iloc[:,-1].values
#Target variable=Income

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=18)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

prediction=model.predict([[45,12]])
print(prediction)

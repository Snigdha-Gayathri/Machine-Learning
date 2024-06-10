import pandas as pd
import numpy as np

a=[i for i in 'abcdefghijklmn']
data=pd.read_csv(r"C:\Users\snigd\OneDrive\Desktop\ML\boston.csv",names=a)

data.dropna(inplace=True)
x=np.array(data.iloc[:,:-1])

y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=2)


##from sklearn.linear_model import LinearRegression
##model=LinearRegression()


##from sklearn.tree import DecisionTreeRegressor
##model=DecisionTreeRegressor()


##from sklearn.ensemble import RandomForestRegressor
##model=RandomForestRegressor()


from sklearn.svm import SVR
model=SVR()

model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

from sklearn.metrics import mean_squared_error
import math
print(math.sqrt(mean_squared_error(ytest,ypred)))



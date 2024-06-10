import pandas as pd
import numpy as np


a=[i for i in 'abcdefghijklmno']
data=pd.read_csv(r"C:\Users\snigd\OneDrive\Desktop\ML\winesdata.csv",names=a)

data.dropna(inplace=True)
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=4)



from sklearn.tree import DecisionTreeClassifier
model=DecisonTreeClassifier()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytest)*100)


##print(model.predict([[1,13.41,3.84,2.12,18.8,90,2.45,2.68,0.27,1.48,4.28,0.91,3,1035]])

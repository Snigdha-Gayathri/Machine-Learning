import pandas as pd
import numpy as np
import matplotlib as mat


##data=pd.read_csv(r"C:\Users\snigd\OneDrive\Desktop\SEMESTER-III\iris.csv") ## Reading the file from wherever it has been saved
####print(data.head()) ##prints the first five rows
##
##
####data.fillna(1.2) ##Filling empty rows with a suitable value
##data.dropna(inplace=True) ##Deletes empty ROWS by default and places the edited data back into 'data'
##
####
##x=np.array(data.iloc[:,:-1]) ## Converts the dataset into an array of arrays.
#### The main array represents number of rows-150. The sub arrays represent number of coulmns in each row-4
####All the 150 rows and total number of columns-1,the last column is not printed
##y=data.iloc[:,-1].values ##column values except the last column are printed
####print(x)

##
##from sklearn.model_selection import train_test_split ##imports the train and test module from sklearn
##xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=4) ##The values of xtest,xtrain,ytest,ytrain are allocated by train_test_split.
####0.20 that is 20% of the data,i.e.,30 rows are tested and 80%,i.e.,120 rows are trained.
####The randomness of the train and test method is determined by the random_state which can be anything.
##
##


##from sklearn.neighbors import KNeighborsClassifier ##Imports KNN algorithm 
##model=KNeighborsClassifier(n_neighbors=3) ##Imports KNN algorithm as model
## ## K value considered is 3
##
##model.fit(xtrain,ytrain) ##The complete gist of the KNN algorithm
##
##
##ypred=model.predict(xtest) ## Predicts the output->ypred whicn favorably need to coincide with ytest.
##Input for prredicting ypred is xtest whose output originally,from the dataset is,ytest.


##from sklearn.metrics import accuracy_score ## Imports accuracy score from metrics package in sklearn
##print(accuracy_score(ytest,ypred)*100) ## predicts the accuracy score taking ytest and ypred as two arguments.


a=[i for i in 'abcdefghijklmn']
data=pd.read_csv(r"C:\Users\snigd\OneDrive\Desktop\boston.csv",names=a)
x=np.array(data.iloc[:,:-1])
y=data.iloc[:,-1].values
print(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=2)
from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(xtrain,ytrain)

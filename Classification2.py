import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_digits
dataset = load_digits()

df=pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

x=df
y=dataset.target


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.2,random_state=9)

#from sklearn.tree import DecisionTreeClassifier
#model=DecisionTreeClassifier() #Iski acuracy hai-->86.11111111111111%

from sklearn.linear_model import  LogisticRegression
model=LogisticRegression() #Iski accuracy-->96.11111111111111%

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

min_vals = df.min().values
max_vals = df.max().values
random_row = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]


#print(random_row) 

prediction=model.predict([[0.0, 6.411647809133279, 0.3434235847336691, 7.91591340181143, 10.157584403418129, 10.050940119264927, 12.932617323589183, 0.31292433998224456, 1.1503601521008815, 15.00019328177568, 8.526813973673953, 10.757466019639626, 4.129847628574073, 9.953064988728665, 10.091830765353333, 2.3776624316233597, 1.1768143639022997, 8.856600343889525, 7.426959862621285, 6.0974948715247095, 13.428191675197048, 12.175836617135607, 10.669175765639126, 4.504717879902228, 0.5644603306171635, 5.617483543738523, 2.0360409524842833, 3.6329737117602097, 1.0949495807708942, 12.93621806747622, 9.672163351976831, 0.17496734370777955, 0.0, 0.5124697646394123, 12.355221660417898, 4.620137808251176, 13.15421121553252, 2.5097093583729677, 3.0850121855143313, 0.0, 0.2837212810245453, 5.594813522495224, 5.936535241228569, 11.81694151251919, 1.7986617685169417, 6.611215544321354, 4.4265206744548, 1.4440973782663873, 7.43620495515422, 10.762370287336138, 3.34076146157979, 3.824407861393686, 3.8503636452015506, 13.001320046553824, 12.464841212629766, 3.07285218218692, 0.7860662088733681, 7.264673515306087, 7.910504875699051, 10.409124971515869, 2.86673180909397, 12.472938810501466, 10.716449217084758, 11.7719191293407]])

print(prediction)

#[0] represents the digit 0
#[1] represents the digit 1
#[2] represents the digit 2
#[3] represents the digit 3
#[4] represents the digit 4
#[5] represents the digit 5
#[6] represents the digit 6
#[7] represents the digit 7
#[8] represents the digit 8
#[9] represents the digit 9

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

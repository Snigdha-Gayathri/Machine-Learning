import pandas as pd 
from sklearn.datasets import load_digits 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt


digits=load_digits()

print("Image data's shape: ",digits.data.shape) 
print("Label data's shape: ",digits.target.shape) 

plt.figure(figsize=(20,4)) 
for index,(image,label) in enumerate(zip(digits.data[0:9],digits.target[0:9])): 
    plt.subplot(1,9,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n'%label,fontsize=20)

    
from sklearn.model_selection import train_test_split  
xtrain,xtest,ytrain,ytest=train_test_split(digits.data,digits.target,test_size=0.20,random_state=4)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)

print(model.predict(xtest[0].reshape(1,-1)))

print(model.predict(xtest[0:10]))

ypred = model.predict(xtest)


from sklearn.metrics import r2_score
score=model.score(xtest,ytest)
print(score)


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)

print("Confusion Matrix:")
print(cm)

print(digits.target.shape)
print(ypred.shape)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

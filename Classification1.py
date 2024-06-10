import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random                       
from sklearn.datasets import load_breast_cancer 
cancerdataset = load_breast_cancer() 


print(cancerdataset.data.shape)  
print(cancerdataset.target.shape) 
print(cancerdataset.DESCR) 
print(cancerdataset.feature_names) 
print(cancerdataset.target_names)



df = pd.DataFrame(data=cancerdataset.data, columns=cancerdataset.feature_names) 


x=df

y=cancerdataset.target 


from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=44)

from sklearn.tree import DecisionTreeClassifier   
model = DecisionTreeClassifier()

  
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

min_vals = df.min().values 
max_vals = df.max().values 
random_row = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_vals, max_vals)]


# Print the single row values
print(random_row) 

prediction=model.predict([[9.727508721597378, 12.229242565370203, 109.71637389066117, 1027.267720250023, 0.15454185152041222, 0.3044384628706409, 0.3013666129703438, 0.01043815351157017, 0.2155435677322765, 0.09675183215379798, 2.258491506808104, 3.7813937264245516, 21.717534402882514, 514.7946425276349, 0.0076355254787153935, 0.016483645322376023, 0.19201402449240662, 0.03508585583098205, 0.06535655913376087, 0.026714480230838774, 18.33079303127448, 29.72785314140913, 236.130220284913, 450.45760777266884, 0.10680822616131776, 0.9716735736044274, 0.23514085247251337, 0.005716499325935789, 0.17808441608772815, 0.08616614918705096]])


print(prediction)

 #For this dataset, 0--> Malignant and 1-->Benign

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)




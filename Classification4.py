import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random                       
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.data"
dataset = pd.read_csv(url, header=None)



df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names) 


x=df

y=dataset.target 


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



#print(prediction)

 #For this dataset, 0--> Malignant and 1-->Benign

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

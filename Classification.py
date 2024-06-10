import pandas as pd
import numpy as np


from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
cancer_data = load_breast_cancer()

# Extract features (X) and target variable (y)
x = cancer_data.data  # Features
y = cancer_data.target  # Target variable

# Print dataset information
#print("Feature names:", cancer_data.feature_names)
#print("Target names:", cancer_data.target_names)
#print("Number of samples:", len(X))
#print("Number of features:", len(cancer_data.feature_names))
#print("Classes:", cancer_data.target_names)


df = pd.DataFrame(cancer_data)
x = df[['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']]  
y = df['malignant' 'benign']


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier   
model = DecisionTreeClassifier()

  
model.fit(xtrain, ytrain)


ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,ypred)*100)




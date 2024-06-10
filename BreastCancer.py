from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib as plt
import pandas as pd
import numpy as np
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=42)

model = DecisionTreeClassifier(random_state=0)

model.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))


model = DecisionTreeClassifier(max_depth=4, random_state=0)

model.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))


from sklearn.tree import export_graphviz
export_graphviz(model, out_file="model.dot", class_names=["malignant", "benign"],
 feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz
with open("model.dot") as f:
 dot_graph = f.read()
graphviz.Source(dot_graph)

print("Feature importances:\n{}".format(model.feature_importances_))
def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
## plt.bar(range(n_features), model.feature_importances_, align='center')
## plt.yticks(np.arange(n_features), cancer.feature_names)
## plt.xlabel("Feature importance")
## plt.ylabel("Feature")
##plot_feature_importances_cancer(model)



# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset, replace 'your_dataset.csv' with the actual dataset file
data = pd.read_csv(r"C:\Users\snigd\OneDrive\Desktop\ML\boston.csv")

data.dropna(inplace=True)

# Extract the independent variable(s) (features) and dependent variable (target)
# For example, let's assume you have one independent variable 'X' and one dependent variable 'y'.
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Make predictions on the test data
ypred = model.predict(xtest)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(ytest,ypred)
mse = mean_squared_error(ytest, ypred)
error_percentage = (mae / np.mean(ytest)) * 100 

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Error Percentage:", error_percentage, "%")







predicted_value = model.predict([[1.05393, 0, 8.14,  0,  0.538,  5.935,  29.3,  4.4986,  4,  307,  21,   386.85,  6.58]])
print(predicted_value)




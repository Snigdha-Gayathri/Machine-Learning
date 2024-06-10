import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv(r'C:\Users\snigd\ML\Laptop_price.csv')


encoder = OneHotEncoder()


brand_name_2D = data['Brand'].values.reshape(-1, 1)


one_hot_encoded = encoder.fit_transform(brand_name_2D)
one_hot_encoded = encoder.fit_transform(brand_name_2D).toarray()



encoded_features_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())

# Concatenate the encoded features with the rest of the dataset
# Make sure to drop the original 'brand_name' column from the dataset
X = pd.concat([data.drop('Brand', axis=1), encoded_features_df], axis=1)


y = data.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=18)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

 
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)

# Calculate Root Mean Squared Error
rmse = sqrt(mse)
print('Root Mean Squared Error: ', rmse)

from sklearn.metrics import r2_score

# Assuming y_test are the true values and y_pred are the predictions made by your model
r_squared = r2_score(y_test, y_pred)
print('R-squared Score: ', r_squared)

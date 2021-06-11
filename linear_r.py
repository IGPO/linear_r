# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 

# Reading training dataset
print('Reading training dataset')
file = "input_data/internship_train.csv"
dataset = pd.read_csv(file)

y = dataset.target          # target
X = dataset.iloc[:,  :-1]   # features

# Splitting dataset
print('Splitting dataset')
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state = 1)

# Defining model
print('Defining model')
my_model = DecisionTreeRegressor(random_state=1)

# Fitting model
print('Fitting model')
my_model.fit(train_X, train_y)

# Make predictions on validation dataset
print('Validating model')
val_y_predictions = my_model.predict(val_X)
print("Root Mean Square Error:\n", np.sqrt(mean_squared_error(val_y, val_y_predictions)))

# Reading dataset for prediction
print('Reading dataset for prediction')
p_file = "input_data/internship_hidden_test.csv"
p_dataset = pd.read_csv(p_file)

# Evaluating predictions
print('Evaluating predictions')
predictions = my_model.predict(p_dataset)

# Saving predictions
print('Saving predictions')
np.savetxt("predictions.csv", predictions, delimiter=",")
print('Done')

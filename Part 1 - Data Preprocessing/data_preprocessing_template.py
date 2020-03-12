# Data Preprocessing Template

# Importing the libraries
import numpy as np #contains mathematical features
import matplotlib.pyplot as plt #drawing charts
import pandas as pd #use for importing and managing datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #feature matrix
y = dataset.iloc[:, 3].values #the expected values

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
Feature scaling reduces the dominating effect that large values may have in 
training models because of the fact that most training model utilize the 
eucledian distance of data points to draw conclusions
"""

#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
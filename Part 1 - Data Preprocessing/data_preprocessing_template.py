# Data Preprocessing Template

# Importing the libraries
import numpy as np #contains mathematical features
import matplotlib.pyplot as plt #drawing charts
import pandas as pd #use for importing and managing datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #feature matrix
y = dataset.iloc[:, 3].values #the expected values


#Taking care of missing data 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imuputer = imputer.fit(X[:, 1:3]) #choose the correct columns in the matrix inorder to fit
X[:, 1:3] =  imputer.transform(X[:, 1:3])


"""
Encoding categorical data mean representing classes fields/columns into
more predictable mathematical format.
it basically to normalize labels.
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,0] = labelencoder_X.fit_transform(X[:,0])

"""
X = array([[0, 44.0, 72000.0],
       [2, 27.0, 48000.0],
       [1, 30.0, 54000.0],
       [2, 38.0, 61000.0],
       [1, 40.0, 63777.77777777778],
       [0, 35.0, 58000.0],
       [2, 38.77777777777778, 52000.0],
       [0, 48.0, 79000.0],
       [1, 50.0, 83000.0],
       [0, 37.0, 67000.0]], dtype=object)
the first column repping country name may make machine learning methods think that there
are some significant relationships between countries. To fix this we need to use
dummy encoding.
"""
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
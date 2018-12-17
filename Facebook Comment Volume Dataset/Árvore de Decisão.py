import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

dataset = pd.read_csv( "data_facebook.csv" , low_memory = False )
dataset.drop(dataset.index[ 0 ], inplace = True )
dataset.drop(dataset.index[ 0 ], inplace = True )
X = dataset[[ 'Feature 11' , 'Feature 26' , 'Feature 16' , 'Feature 36' , 'Feature 35' , 'Feature30' ]].values
y = dataset[ 'Feature 54' ]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2 , random_state =0 )

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print ( 'Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred))
print ( 'Mean Squared Error:' , metrics.mean_squared_error(y_test, y_pred))
print ( 'Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

metrics.r2_score(y_test, y_pred)
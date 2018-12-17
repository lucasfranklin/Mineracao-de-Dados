import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from subprocess import check_output
from datetime import time

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestRegressor(n_jobs =-1 )
estimators = np.arange( 10 , 200 , 10 )
scores = []

for n in estimators:
	model.set_params(n_estimators = n)
	model.fit(X_train, y_train)
	scores.append(model.score(X_test, y_test))
	plt.title( "Effect of n_estimators" )
	plt.xlabel( "n_estimator" )
	plt.ylabel( "score" )
	plt.plot(estimators, scores)
	np.mean(scores)
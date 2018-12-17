import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
from sklearn import metrics

data = pd.read_csv( "Features_Variant_1.csv" )

used_features = [
'Feature 1' , 'Feature 2' , 'Feature 3' , 'Feature 4' , 'Feature 5' ,
'Feature 6' , 'Feature 7' , 'Feature 8' , 'Feature 9' , 'Feature 10' ,
'Feature 11' , 'Feature 12' , 'Feature 13' , 'Feature 14' , 'Feature 15' ,
'Feature 16' , 'Feature 17' , 'Feature 18' , 'Feature 19' , 'Feature 20' ,
'Feature 21' , 'Feature 22' , 'Feature 23' , 'Feature 24' , 'Feature 25' ,
'Feature 26' , 'Feature 27' , 'Feature 28' , 'Feature 29' , 'Feature 30' ,
'Feature 31' , 'Feature 32' , 'Feature 33' , 'Feature 34' , 'Feature 35' ,
'Feature 36' , 'Feature 37' , 'Feature 38' , 'Feature 39' , 'Feature 40' ,
'Feature 41' , 'Feature 42' , 'Feature 43' , 'Feature 44' , 'Feature 45' ,
'Feature 46' , 'Feature 47' , 'Feature 48' , 'Feature 49' , 'Feature 50' ,
'Feature 51' , 'Feature 52' , 'Feature 53'
]

X_train, X_test = train_test_split(data, test_size =0.3 , random_state = int (time.time()))

clf = BayesianRidge()
clf.fit(X_train[used_features].values, X_train[ "Feature 54" ])
y_pred = clf.predict(X_test[used_features])

print ( "Precisão: {:.2%} ".format(clf.score( X_test[used_features].values, X_test[ "Feature54" ])))
print (metrics.r2_score(X_test[ "Feature 54" ],y_pred))
print ( "Cross Validation: {:.2%} ".format(np.mean(cross_val_score(clf, X_test[used_features].values, X_test[ "Feature 54" ], cv =10 ))))
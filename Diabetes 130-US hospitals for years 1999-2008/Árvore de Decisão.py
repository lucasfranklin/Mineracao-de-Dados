import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd . read_csv( "./diabetes_data_preprocessed.csv" )

X = df[[ 'gender' , 'race' , 'discharge_disposition_id' , 'time_in_hospital' ,
'age' , 'num_medications' , 'num_procedures' , 'admission_type_id' ,
'service_utilization' , 'numchange' , 'nummed' ]] . values

y = df[ "readmitted" ]


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size =0.3 , random_state =3 )
drugTree = DecisionTreeClassifier(criterion = "entropy" , max_depth = 8 )
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

from sklearn import metrics
import matplotlib.pyplot as plt

print ( "Precisão: {:.2%} " . format(metrics.accuracy_score(y_testset, predTree)))
print ( "Cross Validation: {:.2%} " . format(np.mean(cross_val_score(predTree,X_trainset, y_trainset, cv =10 ))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

data = pd . read_csv( "data/diabetes_data_preprocessed.csv" )

X_train, X_test = train_test_split(data, test_size =0.2 , random_state = int (time . time()))

used_features = ['encounter_id',
'patient_nbr', 'race', 'gender', 'age', 'admission_type_id' ,
'discharge_disposition_id', 'admission_source_id', 'time_in_hospital' ,
'num_lab_procedures', 'num_procedures', 'num_medications' ,
'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1' ,
'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult' ,
'metformin', 'repaglinide', 'nateglinide' ,
'glimepiride', 'acetohexamide', 'glipizide', 'tolbutamide' ,
'pioglitazone' , 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin' ,
'change', 'diabetesMed', 'readmitted' ,
'service_utilization', 'numchange', 'nummed' ]

clf = svm . SVC() . fit(X_train[used_features] . values, X_train[ "readmitted" ])
print ( "Precisão: {:.2%} " . format(clf.score( X_test[used_features].values, X_test[ "readmitted" ])))
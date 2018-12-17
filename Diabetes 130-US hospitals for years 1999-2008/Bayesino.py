import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score

data = pd.read_csv( "data/diabetes_data_preprocessed.csv" )

used_features = [
'gender' , 'age' , 'admission_type_id' ,
'discharge_disposition_id' , 'admission_source_id' , 'time_in_hospital' ,
'num_lab_procedures' , 'num_procedures' , 'num_medications' ,
'number_outpatient' , 'number_emergency' , 'number_inpatient' , 'diag_1' ,
'diag_2' , 'diag_3' , 'number_diagnoses' , 'max_glu_serum' , 'A1Cresult' ,
'metformin' , 'chlorpropamide' ,
'glimepiride' , 'acetohexamide' , 'glipizide' , 'glyburide' , 'tolbutamide' ,
'pioglitazone' , 'miglitol' , 'troglitazone' ,
'tolazamide' , 'insulin' , 'glyburide-metformin' , 'glipizide-metformin' ,
'glimepiride-pioglitazone' , 'metformin-rosiglitazone' ,
'metformin-pioglitazone' , 'change' , 'diabetesMed' ,
'service_utilization' , 'level1_diag1' ,
'level2_diag1' , 'level1_diag2' , 'level2_diag2' , 'level1_diag3' ,
'level2_diag3']

X_train, X_test = train_test_split(data, test_size =0.2, random_state = int (time.time()))
gnd = GaussianNB()
gnd . fit(X_train[used_features].values,X_train[ "readmitted" ])
y_pred = gnd.predict(X_test[used_features])
print ( "Precisão: {:.2%} " . format(gnd . score( X_test[used_features].values, X_test[ "readmitted" ])))
print ( "Cross Validation: {:.2%} " . format(np . mean(cross_val_score(gnd, X_test[used_features].values, X_test[ "readmitted" ], cv =10 ))))
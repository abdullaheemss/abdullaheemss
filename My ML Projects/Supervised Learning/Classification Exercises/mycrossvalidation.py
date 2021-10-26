#import pandas 
import pandas as pd

#read the data
f = pd.read_csv('AdultIncome.csv')

#split into X and Y
X = f.iloc[:,:-1]
Y = f.iloc[:,-1]

#import the necessary classifiers
#import decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=0.5)

#implement cross validation
from sklearn.model_selection import cross_validate
cross_validate_dtc = cross_validate(dtc, X, Y, cv=10, return_train_score=True)
cross_validate_rfc = cross_validate(rfc, X, Y, cv=10, return_train_score=True)
cross_validate_svc = cross_validate(svc, X, Y, cv=10, return_train_score=True)

#get the average 
import numpy as np
dtc_test_average = np.average(cross_validate_dtc['test_score'])
rfc_test_average = np.average(cross_validate_rfc['test_score'])
svc_test_average = np.average(cross_validate_svc['test_score'])

dtc_train_average = np.average(cross_validate_dtc['train_score'])
rfc_train_average = np.average(cross_validate_rfc['train_score'])
svc_train_average = np.average(cross_validate_svc['train_score'])


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier 
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing 
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
target = dataSetPositive['Is Back'].map({'Yes': 1, 'No': 0})
dataSetPositive = dataSetPositive.apply(LabelEncoder().fit_transform)
featureNames = list(dataSetPositive.columns)
featureNames.remove('Unnamed: 0')
featureNames.remove('Customer ID')
featureNames.remove('Date')
featureNames.remove('Is Back')
scalesFeatureNames = ['Total Quantity', 'Total Price',  'Purchase Count']
scaledFeatures = dataSetPositive[featureNames].copy()
featuresValues = scaledFeatures[scalesFeatureNames]
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
scaler = min_max_scaler.fit(featuresValues.values)
featuresValues = scaler.transform(featuresValues.values)
scaledFeatures[scalesFeatureNames] = featuresValues
trainData, testData, trainTargets, testTargets = train_test_split(scaledFeatures, target, stratify = target, test_size = 0.2, random_state = 10) # 70% training and 30% test 

desisionTreeClassifier = DecisionTreeClassifier(max_depth = 3)
KNNClassifier = KNeighborsClassifier(n_neighbors = 49, weights = 'distance', p =2, n_jobs = -1)
logisticClassifier = LogisticRegression(random_state = 1, max_iter = 1000,  C=3.0)
estimator = []
estimator.append(('DT', desisionTreeClassifier))
estimator.append(('KNN', KNNClassifier))
estimator.append(('LR', logisticClassifier))
votingHardClassifier = VotingClassifier(estimators = estimator, voting ='hard', weights=[3,1,0.5]) 
votingHardClassifier = votingHardClassifier.fit(trainData, trainTargets) 

predictedTargets = votingHardClassifier.predict(testData)
precisionIsBack = metrics.precision_score(testTargets, predictedTargets, pos_label=1)
precisionIsNBack = metrics.precision_score(testTargets, predictedTargets, pos_label=0)
recallIsBack = metrics.recall_score(testTargets, predictedTargets, pos_label=1)
recallIsNBack = metrics.recall_score(testTargets, predictedTargets, pos_label=0)

print("Is Back")
print("Accuracy: ", metrics.accuracy_score(testTargets, predictedTargets) * 100)
print("Percision: ", precisionIsBack * 100)
print("Recall: ", recallIsBack * 100)
print("Is Not Back")
print("Accuracy: ", metrics.accuracy_score(testTargets, predictedTargets) * 100)
print("Percision: ", precisionIsNBack * 100)
print("Recall: ", recallIsNBack * 100)
WsBack =  len(testTargets[testTargets == 1])
WsNBack =  len(testTargets[testTargets == 0])
print("Avarage Precision: ", ((WsBack * precisionIsBack) + (WsNBack * precisionIsNBack)) / (WsBack + WsNBack) * 100)
print("Avarage Recall: ", ((WsBack * recallIsBack) + (WsNBack * recallIsNBack)) / (WsBack + WsNBack) * 100)  
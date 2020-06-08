from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


#neigh = KNeighborsClassifier(n_neighbors=3)
dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
#dataSetPositive["is_weakend"] = np.where(dataSetPositive["Date"].dt.dayofweek.isin([5, 6]), 1, 0)
#d = pd.get_dummies(dataSetPositive['Country'])
#dataSetPositive = dataSetPositive.join(d)
target = dataSetPositive['Is Back'].map({'Yes': 1, 'No': 0})
dataSetPositive = dataSetPositive.apply(LabelEncoder().fit_transform)
featureNames = list(dataSetPositive.columns)
featureNames.remove('Unnamed: 0')
featureNames.remove('Customer ID')
#featureNames.remove('Country')
featureNames.remove('Date')
featureNames.remove('Is Back')
scalesFeatureNames = ['Total Quantity', 'Total Price', 'Purchase Count', 'Country']
scaledFeatures = dataSetPositive[featureNames].copy()
featuresValues = scaledFeatures[scalesFeatureNames]
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
scaler = min_max_scaler.fit(featuresValues.values)
featuresValues = scaler.transform(featuresValues.values)
scaledFeatures[scalesFeatureNames] = featuresValues
trainData, testData, trainTargets, testTargets = train_test_split(scaledFeatures, target, test_size = 0.2, random_state = 10) # 70% training and 30% test
KNNClassifier = KNeighborsClassifier(n_neighbors = 49)#, weights = 'distance', p =2, n_jobs = -1)
baggibgClassifier = BaggingClassifier(base_estimator=KNNClassifier, n_estimators=15, max_samples = 0.5,
        max_features = 0.5, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=11, verbose=0)

baggibgClassifier = baggibgClassifier.fit(trainData, trainTargets)
predictedTargets = baggibgClassifier.predict(testData)

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
print("CHECK:")
print("Avarage Precision: ", ((WsBack * precisionIsBack) + (WsNBack * precisionIsNBack)) / (WsBack + WsNBack) * 100)
print("Avarage Recall: ", ((WsBack * recallIsBack) + (WsNBack * recallIsNBack)) / (WsBack + WsNBack) * 100)  
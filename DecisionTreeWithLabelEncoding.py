from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
scalesFeatureNames = ['Total Quantity', 'Total Price',  'Purchase Count']
scaledFeatures = dataSetPositive[featureNames].copy()
featuresValues = scaledFeatures[scalesFeatureNames]
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
scaler = min_max_scaler.fit(featuresValues.values)
featuresValues = scaler.transform(featuresValues.values)
scaledFeatures[scalesFeatureNames] = featuresValues
trainData, testData, trainTargets, testTargets = train_test_split(scaledFeatures, target, test_size = 0.2, random_state = 10) # 70% training and 30% test 
desisionTreeClassifier = DecisionTreeClassifier(max_depth = 3)
desisionTreeClassifier = desisionTreeClassifier.fit(trainData, trainTargets)
predictedTargets = desisionTreeClassifier.predict(testData)
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
print("Confusion Matrix: \n", metrics.confusion_matrix(testTargets, predictedTargets)) 

#Plotting;
depthList = []
accuracyTestList = []
precisionTestList = []
recallTestList = []
accuracyTrainList = []
precisionTrainList = []
recallTrainList = []
for i in range(1, 20):
    depthList.append(i)
    desisionTreeClassifierPlotting = DecisionTreeClassifier(max_depth = i)
    desisionTreeClassifierPlotting = desisionTreeClassifierPlotting.fit(trainData, trainTargets)
    #Test Data
    predictedTargets = desisionTreeClassifierPlotting.predict(testData)
    accuracy = metrics.accuracy_score(testTargets, predictedTargets) * 100
    accuracyTestList.append(accuracy)
    precision = metrics.precision_score(testTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100
    precisionTestList.append(precision)
    recall = metrics.recall_score(testTargets, predictedTargets, average = 'weighted',  labels=np.unique(predictedTargets)) * 100
    recallTestList.append(recall)

    #Train Data
    predictedTargets = desisionTreeClassifierPlotting.predict(trainData)
    accuracy = metrics.accuracy_score(trainTargets, predictedTargets) * 100
    accuracyTrainList.append(accuracy)
    precision = metrics.precision_score(trainTargets, predictedTargets, average = 'weighted',  labels=np.unique(predictedTargets)) * 100
    precisionTrainList.append(precision)
    recall = metrics.recall_score(trainTargets, predictedTargets, average = 'weighted',  labels=np.unique(predictedTargets)) * 100
    recallTrainList.append(recall)

plt.plot(depthList, accuracyTestList, label = 'Test Data')
plt.plot(depthList, accuracyTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Accuracy')
plt.title('Accuracy of Decision Tree with different max_depth')
plt.legend(loc="center")

plt.figure()
plt.plot(depthList, precisionTestList, label = 'Test Data')
plt.plot(depthList, precisionTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Average Precision')
plt.title('Average precision of Decision Tree with different max_depth')
plt.legend(loc="center")

plt.figure()
plt.plot(depthList, recallTestList, label = 'Test Data')
plt.plot(depthList, recallTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Average Recall')
plt.title('Average recall of Decision Tree with different max_depth')
plt.legend(loc="center")

plt.show()

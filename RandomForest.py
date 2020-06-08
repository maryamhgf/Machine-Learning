from sklearn.ensemble import RandomForestClassifier
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


def randomOversampling(imbalancedData, imbalancedTargets):
    print("INSIDE :::::::::::::::::::::::;;;")
    imbalanced = imbalancedData.join(imbalancedTargets)
    IsBackData = imbalanced.loc[imbalanced['Is Back'] == 1]
    IsNotBackData = imbalanced.loc[imbalanced['Is Back'] == 0]
    oneLabels = len(imbalancedTargets[imbalancedTargets == 1])
    zeroLabels = len(imbalancedTargets[imbalancedTargets == 0])
    new = pd.DataFrame()
    if(oneLabels > zeroLabels):
        copyNumber = int(oneLabels / zeroLabels)
        for i in range(copyNumber):
            new = new.append(IsNotBackData)
        new = new.append(IsBackData)
        remaining = 2 * oneLabels - len(new)
        new = new.append(IsNotBackData.sample(n = remaining))
    if(zeroLabels >= oneLabels):
        copyNumber = int(zeroLabels / oneLabels)
        for i in range(copyNumber):
            new = new.append(IsBackData)
        new = new.append(IsNotBackData)
        remaining = zeroLabels - len(new)
        new = new.append(IsNotBackData.sample(n = remaining))
    #print(shuffle(new.drop(['Is Back'], axis=1)))
    return new['Is Back'], new.drop(['Is Back'], axis=1)

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

randomForestClassifier = RandomForestClassifier(n_estimators=60, max_depth=5,max_features=0.5, max_samples=0.5, random_state=11, min_samples_leaf=1 , max_leaf_nodes = 10)
randomForestClassifier = randomForestClassifier.fit(trainData, trainTargets)
predictedTargets = randomForestClassifier.predict(testData)
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

#Plotting:
#Max_Depth:
depthList = []
accuracyTrainList = []
precisionTrainList = []
recallTrainList = []
accuracyTestList = []
precisionTestList = []
recallTestList = []

for i in range(1, 20):
    depthList.append(i)
    randomForestClassifier = RandomForestClassifier(n_estimators=60, max_depth=i,max_features=0.5, max_samples=0.5, random_state=11, min_samples_leaf=1 , max_leaf_nodes = 10)
    randomForestClassifier = randomForestClassifier.fit(trainData, trainTargets)
    #Test Data:
    predictedTargets = randomForestClassifier.predict(testData)
    accuracyTestList.append(metrics.accuracy_score(testTargets, predictedTargets) * 100)
    precisionTestList.append(metrics.precision_score(testTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    recallTestList.append(metrics.recall_score(testTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    #Train Data:
    predictedTargets = randomForestClassifier.predict(trainData)
    accuracyTrainList.append(metrics.accuracy_score(trainTargets, predictedTargets) * 100)
    precisionTrainList.append(metrics.precision_score(trainTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    recallTrainList.append(metrics.recall_score(trainTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)  

plt.plot(depthList, accuracyTestList, label = 'Test Data')
plt.plot(depthList, accuracyTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Accuracy')
plt.title('Accuracy of Decision Tree with different max_depth(n_estimators=60,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")

plt.figure()
plt.plot(depthList, precisionTestList, label = 'Test Data')
plt.plot(depthList, precisionTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Average Precision')
plt.title('Average precision of Decision Tree with different max_depth(n_estimators=60,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")

plt.figure()
plt.plot(depthList, recallTestList, label = 'Test Data')
plt.plot(depthList, recallTrainList, label = 'Train Data')
plt.xlabel('Maximum Depth(int)')
plt.ylabel('Average Recall')
plt.title('Average recall of Decision Tree with different max_depth(n_estimators=60,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")


#Changing n_estimator:
numberList = []
accuracyTrainList = []
precisionTrainList = []
recallTrainList = []
accuracyTestList = []
precisionTestList = []
recallTestList = []
for i in range(1, 300, 10):
    numberList.append(i)
    randomForestClassifier = RandomForestClassifier(n_estimators=i, max_depth=5,max_features=0.5, max_samples=0.5, random_state=11, min_samples_leaf=1 , max_leaf_nodes = 10)
    randomForestClassifier = randomForestClassifier.fit(trainData, trainTargets)
    #Test Data:
    predictedTargets = randomForestClassifier.predict(testData)
    accuracyTestList.append(metrics.accuracy_score(testTargets, predictedTargets) * 100)
    precisionTestList.append(metrics.precision_score(testTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    recallTestList.append(metrics.recall_score(testTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    #Train Data:
    predictedTargets = randomForestClassifier.predict(trainData)
    accuracyTrainList.append(metrics.accuracy_score(trainTargets, predictedTargets) * 100)
    precisionTrainList.append(metrics.precision_score(trainTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)
    recallTrainList.append(metrics.recall_score(trainTargets, predictedTargets, average = 'weighted', labels=np.unique(predictedTargets)) * 100)  

plt.figure()
plt.plot(numberList, accuracyTestList, label = 'Test Data')
plt.plot(numberList, accuracyTrainList, label = 'Train Data')
plt.xlabel('n_estimator')
plt.ylabel('Accuracy')
plt.title('Accuracy of Decision Tree with different n_estimators(max_depth=5,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")

plt.figure()
plt.plot(numberList, precisionTestList, label = 'Test Data')
plt.plot(numberList, precisionTrainList, label = 'Train Data')
plt.xlabel('n_estimator')
plt.ylabel('Average Precision')
plt.title('Average precision of Decision Tree with different n_estimators(max_depth=5,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")

plt.figure()
plt.plot(numberList, recallTestList, label = 'Test Data')
plt.plot(numberList, recallTrainList, label = 'Train Data')
plt.xlabel('n_estimator')
plt.ylabel('Average Recall')
plt.title('Average recall of Decision Tree with different n_estimators(max_depth=5,max_features=0.5, max_samples=0.5, min_samples_leaf=1 , max_leaf_nodes = 10)')
plt.legend(loc="center")

plt.show()
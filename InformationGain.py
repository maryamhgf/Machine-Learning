import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt; plt.rcdefaults()

pd.options.mode.chained_assignment = None

#With One Hot Encoding For All Datas:
dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
dataSetPositive["Is W"] = np.where(dataSetPositive["Date"].dt.dayofweek.isin([5, 6]), 1, 0)
d = pd.get_dummies(dataSetPositive['Country'])
dataSetPositive = dataSetPositive.join(d)
featureNames = list(dataSetPositive.columns)
featureNames.remove('Unnamed: 0')
featureNames.remove('Country')
featureNames.remove('Date')
featureNames.remove('Is Back')
featuresData = dataSetPositive[featureNames]
features = featureNames
scaled_features = featuresData.copy()
col_names = ['Total Quantity', 'Total Price', 'Purchase Count']
featuresValues = scaled_features[col_names]
scaler = StandardScaler().fit(featuresValues.values)
featuresValues = scaler.transform(featuresValues.values)
scaled_features[col_names] = featuresValues
labels = dataSetPositive["Is Back"]
labels = labels.replace('Yes', 1)
labels = labels.replace('No', 0)
indexOfCont = features.index('Total Price')
idx = [i for i in range(len(features))]
idx.remove(indexOfCont)
res = dict(zip(features, mutual_info_classif(scaled_features, labels, discrete_features=idx)))
print("Information Gain of each feature to Is Bck feature(For All Data with One Hot Encoding):\n", res)

names = list(dataSetPositive.columns)
objects = featureNames
y_pos = np.arange(len(objects))
performance = list(res.values())
plt.bar(y_pos, performance, align='center', alpha=0.8, width = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Information Gain percantage')
plt.title('Information Gain of each feature to Is Bck feature(For All Data with One Hot Encoding)')
plt.show()
#With Label Encoding For All Datas:
dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
dataSetPositive["Is W"] = np.where(dataSetPositive["Date"].dt.dayofweek.isin([5, 6]), 1, 0)
dataSetPositive = dataSetPositive.apply(LabelEncoder().fit_transform)
featureNames1 = list(dataSetPositive.columns)
featureNames1.remove('Unnamed: 0')
featureNames1.remove('Date')
featureNames1.remove('Is Back')
featuresData1 = dataSetPositive[featureNames1]
features1 = featureNames1
scaled_features1 = featuresData1.copy()
col_names = ['Total Quantity', 'Total Price', 'Purchase Count']
featuresValues1 = scaled_features1[col_names]
scaler1 = StandardScaler().fit(featuresValues1.values)
featuresValues1 = scaler1.transform(featuresValues1.values)
scaled_features1[col_names] = featuresValues1
indexOfCont = features1.index('Total Price')
idx = [i for i in range(len(features1))]
idx.remove(indexOfCont)
res = dict(zip(features1, mutual_info_classif(scaled_features1, labels, discrete_features=idx)))
print("Information Gain of each feature to Is Bck feature(For all Data and with Label Encoding): \n", res)

objects = featureNames1
y_pos = np.arange(len(objects))
performance = list(res.values())
plt.bar(y_pos, performance, align='center', alpha=0.8, width = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Information Gain percantage')
plt.title('Information Gain of each feature to Is Bck feature(For all Data and with Label Encoding)')
plt.show()
#With One Hot Encoding For Train Data:
dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
dataSetPositive["Is W"] = np.where(dataSetPositive["Date"].dt.dayofweek.isin([5, 6]), 1, 0)
d = pd.get_dummies(dataSetPositive['Country'])
dataSetPositive = dataSetPositive.join(d)
featureNames = list(dataSetPositive.columns)
featureNames.remove('Unnamed: 0')
featureNames.remove('Country')
featureNames.remove('Date')
featureNames.remove('Is Back')
featuresData = dataSetPositive[featureNames]
features = featureNames
scaled_features = featuresData.copy()
col_names = ['Total Quantity', 'Total Price', 'Purchase Count']
featuresValues = scaled_features[col_names]
scaler = StandardScaler().fit(featuresValues.values)
featuresValues = scaler.transform(featuresValues.values)
scaled_features[col_names] = featuresValues
labels = dataSetPositive["Is Back"]
labels = labels.replace('Yes', 1)
labels = labels.replace('No', 0)
trainData, testData, trainTargets, testTargets = train_test_split(scaled_features, labels, test_size = 0.2, random_state = 10) 
indexOfCont = features.index('Total Price')
idx = [i for i in range(len(features))]
idx.remove(indexOfCont)
res = dict(zip(features, mutual_info_classif(trainData, trainTargets, discrete_features=idx)))
print("Information Gain of each feature to Is Bck feature(For Train Data and with One Hot Encoing): \n", res)

names = list(dataSetPositive.columns)
objects = featureNames
y_pos = np.arange(len(objects))
performance = list(res.values())
plt.bar(y_pos, performance, align='center', alpha=0.8, width = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Information Gain percantage')
plt.title('Information Gain of each feature to Is Bck feature(For Train Data and with One Hot Encoing)')
plt.show()

#With Label Encoding For Train Datas:
dataSet = pd.read_csv("/home/maryam/Documents/AI/CA4/data.csv")
dataSetPositive = dataSet[(dataSet[["Total Quantity", "Total Price", "Purchase Count"]] > 0).all(1)]
dataSetPositive["Date"] = pd.to_datetime(dataSetPositive["Date"])
dataSetPositive['month'] = dataSetPositive['Date'].dt.month
dataSetPositive["Day"] = dataSetPositive["Date"].dt.dayofweek
dataSetPositive["Is W"] = np.where(dataSetPositive["Date"].dt.dayofweek.isin([5, 6]), 1, 0)
dataSetPositive = dataSetPositive.apply(LabelEncoder().fit_transform)
featureNames1 = list(dataSetPositive.columns)
featureNames1.remove('Unnamed: 0')
featureNames1.remove('Date')
featureNames1.remove('Is Back')
featuresData1 = dataSetPositive[featureNames1]
features1 = featureNames1
scaled_features1 = featuresData1.copy()
col_names = ['Total Quantity', 'Total Price', 'Purchase Count']
featuresValues1 = scaled_features1[col_names]
scaler1 = StandardScaler().fit(featuresValues1.values)
featuresValues1 = scaler1.transform(featuresValues1.values)
scaled_features1[col_names] = featuresValues1
trainData, testData, trainTargets, testTargets = train_test_split(scaled_features1, labels, test_size = 0.2, random_state = 10) 
indexOfCont = features1.index('Total Price')
idx = [i for i in range(len(features1))]
idx.remove(indexOfCont)
res = dict(zip(features1, mutual_info_classif(trainData, trainTargets, discrete_features=idx)))
print("Information Gain of each feature to Is Bck feature(For train Data With One Hot Encoding): \n", res)
objects = featureNames1
y_pos = np.arange(len(objects))
performance = list(res.values())
plt.bar(y_pos, performance, align='center', alpha=0.8, width = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Information Gain percantage')
plt.title('Information Gain of each feature to Is Bck feature(For train Data With One Hot Encoding)')
plt.show()

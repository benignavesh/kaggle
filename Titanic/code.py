import pandas as pd

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')


labels_train = dataset_train.iloc[:,[1]].values
features_train = dataset_train.iloc[:,[2,4,5,6,7]].values

features_test = dataset_test.iloc[:,[1,3,4,5,6]].values


#Missing data analysis for age
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'median',axis = 0)

imputer = imputer.fit(features_train[:,[2]])
features_train[:,[2]] = imputer.transform(features_train[:,[2]])

imputer = imputer.fit(features_test[:,[2]])
features_test[:,[2]] = imputer.transform(features_test[:,[2]])

# Converting Male/ female to 1/0
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
features_train[:,1] = labelencoder.fit_transform (features_train[:,1])
features_test[:,1] = labelencoder.fit_transform (features_test[:,1])


#OneHotEncoding Pclass
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features_train = onehotencoder.fit_transform(features_train).toarray()
features_test = onehotencoder.fit_transform(features_test).toarray()


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)

df = pd.DataFrame(labels_pred)
df.to_csv("submission.csv")
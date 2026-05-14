import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
# Getting rid of duplicates
dataset = dataset.drop_duplicates()
# print(dataset.duplicated().sum())

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scaling the features:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

newData = [[30,87000]]
newData = scaler.transform(newData)

if classifier.predict(newData) == 1:
    print("Yes")
else:
    print("No")


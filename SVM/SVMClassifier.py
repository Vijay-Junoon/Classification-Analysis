import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)
classifier = SVC(kernel="rbf")
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
grid = GridSearchCV(classifier,X_train,cv=10)
print(grid)


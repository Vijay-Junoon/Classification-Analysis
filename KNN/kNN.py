import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X,y)
print(classifier.predict([[30,87000]]))


import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../fast_fullsearch/train_data.csv", sep=",", header=None, encoding="utf-8")

y = data[36]
X = data.drop(36, axis=1)

clf = RandomForestClassifier(n_estimators=100, random_state=22)

clf = clf.fit(X, y)

s = pickle.dumps(clf)
with open("clf.model", 'wb') as fout:
    fout.write(s)
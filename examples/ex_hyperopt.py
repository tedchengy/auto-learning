import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, partial

data = pd.read_csv('/home/chen/桌面/creditcard_01.csv')
targets = data['Class']
data.drop('Class', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.30, random_state=1)
'''
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy score: ", metrics.accuracy_score(y_test, predicted))
print("roc_auc_score:\n ", metrics.roc_auc_score(y_test, predicted))
'''
space = {"max_depth": hp.choice("max_depth", range(1, 3)),
         "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 3)),
         "n_estimators": hp.choice("n_estimators", range(5, 10)), }



def percept(args):
    global X_train, X_test, y_train, y_test
    clf_p = RandomForestClassifier(max_depth=args["max_depth"], n_estimators=args["n_estimators"],
                                   min_samples_leaf=args["min_samples_leaf"], random_state=0)  # 参数搜索
    clf_p.fit(X_train, y_train)
    y_pred = clf_p.predict(X_test)
    return metrics.roc_auc_score(y_test, y_pred)


algo = partial(tpe.suggest, n_startup_jobs=1)

best = fmin(percept, space, algo=algo, max_evals=1)

print(best)


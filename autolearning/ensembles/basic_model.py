from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import pandas as pd
import numpy as np


class IncrementalModelBase:
    def __init__(self, loss='hinge', penalty='l2', l1_ratio=0.15, epsilon=0.1):
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.estimator = None
        self.random_state = 0
        self.sample_weight = 5000  # 在一轮迭代中的批处理量
        self.sample_index = 3  # 数据总共迭代次数

    def sampling(self, X, y):
        s = self.sample_weight
        loop = int(X.shape[0] / s) + 1
        for i in range(loop):
            yield X[i * s:(i + 1) * s], y[i * s:(i + 1) * s]

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X, )

    def fit(self, X, y, **kwargs):
        pass

    def score(self, X, label):
        pass


class IncrementalClassifierModel(IncrementalModelBase):
    def __init__(self, loss='hinge', penalty='l2', l1_ratio=0.15, epsilon=0.1):
        self.classes = None
        super(IncrementalClassifierModel, self).__init__(loss=loss, penalty=penalty, l1_ratio=l1_ratio, epsilon=epsilon)

    def fit(self, X, y, **kwargs):
        estimator = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            epsilon=self.epsilon,
            random_state=self.random_state,
            max_iter=1000,
            tol=1e-3,
        )

        if self.classes is None:
            self.classes = np.unique(y)

        for index in range(self.sample_index):
            for xpi, ypi in self.sampling(X, y):
                estimator.partial_fit(xpi, ypi, classes=self.classes)

        self.estimator = estimator
        return self

    def score(self, X, label):
        y = self.estimator.predict(X)
        num = len(self.classes)

        if num == 2:
            return metrics.roc_auc_score(label, y)
        else:
            return metrics.accuracy_score(label, y)


class IncrementalRegressorModel(IncrementalModelBase):
    def __init__(self, loss='squared_loss', penalty='l2', l1_ratio=0.15, epsilon=0.1):
        super(IncrementalRegressorModel, self).__init__(loss=loss, penalty=penalty, l1_ratio=l1_ratio, epsilon=epsilon)

    def fit(self, X, y, **kwargs):
        estimator = SGDRegressor(
            loss=self.loss,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            epsilon=self.epsilon,
            random_state=self.random_state,
            max_iter=1000,
            tol=1e-3,
        )

        for index in range(self.sample_index):
            for xpi, ypi in self.sampling(X, y):
                estimator.partial_fit(xpi, ypi)

        self.estimator = estimator
        return self

    def score(self, X, label):
        y = self.estimator.predict(X)
        return metrics.r2_score(label, y)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_boston
    import time

    random_state = 0

    data = load_iris()
    target = data.target
    data = data.data

    time_a1 = time.time()
    model_a = SGDClassifier(max_iter=1000, tol=1e-3, random_state=random_state)
    model_a.fit(data, target)
    label_a = model_a.predict(data)
    time_a2 = time.time()
    print(metrics.accuracy_score(target, label_a))
    print(round((time_a2 - time_a1), 5))

    data = pd.read_csv('/r2/data/creditcard_01.csv')
    target = data['Class']
    data.drop('Class', axis=1, inplace=True)

    time_b1 = time.time()
    model_b = IncrementalClassifierModel()
    model_b.fit(data, target)
    label_b = model_b.predict(data)
    time_b2 = time.time()
    print(metrics.accuracy_score(target, label_b))
    print(metrics.roc_auc_score(target, label_b))
    print(round((time_b2 - time_b1), 5))

    data = load_boston()
    target = data.target
    data = data.data

    time_c1 = time.time()
    model_c = IncrementalRegressorModel()
    model_c.fit(data, target)
    label_c = model_c.predict(data)
    time_c2 = time.time()
    print(model_c.score(data, target))
    print(round((time_c2 - time_c1), 5))

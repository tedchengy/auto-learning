from autolearning.ensembles.ensemble_selection import *
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval, partial
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class ClassifierParameterOptimization:
    def __init__(self, data, target, ):

        self.data = data
        self.target = target
        self.columns = data.shape[-1]

        self.n_startup_jobs = 20
        self.best_score = 0
        self.count = 0
        self.space = {}  # 模型的参数空间
        self.best = {}  # hyperopt 直接返回的最优参数空间
        self.best_space = {}  # hyperopt 转换后返回的最优参数空间
        self.best_loss_result = 0
        self.trials = None
        self.estimator = None

    def classifier_f(self, params):
        # print('####', params)
        self.count += 1

        data_ = self.data[:]
        if 'preprocess' in params:
            if params['preprocess'] == 'normalize':
                data_ = preprocessing.normalize(data_)
            elif params['preprocess'] == 'scale':
                data_ = preprocessing.scale(data_)
            elif params['preprocess'] == 'pca':
                data_ = PCA().fit_transform(data_)
            elif params['preprocess'] == 'lda':
                data_ = LinearDiscriminantAnalysis().fit(data_, self.target).transform(data_)
            del params['preprocess']

        try:
            clf = IncrementalClassifierModel(**params).fit(data_, self.target)
        except Exception as e:
            print('e:', e, 'params:', params)
            score = 0
        else:
            # label = clf.predict(self.data)
            # score = metrics.roc_auc_score(label, self.target)
            score = clf.score(data_, self.target)

        if score > self.best_score:
            # print('new best:', score, 'using:', params)
            self.best_score = score
            self.estimator = clf
        if self.count % 100 == 0:
            print('iters:', self.count, 'score:', score, 'using', params)
        return {'loss': -score, 'status': STATUS_OK}

    def transform_space(self, ):
        self.best_space = space_eval(self.space, self.best)
        return self.best_space

    def search_space(self, ):
        self.space = {
            'loss': hp.choice('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                       'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']),
            'l1_ratio': hp.uniform('l1_ratio', 0, 1),
            'epsilon': hp.uniform('epsilon', 0, 1),

            'preprocess': hp.choice('preprocess', [None, 'normalize', 'scale', 'pca', 'lda']),
        }
        return self.space

    def function_min(self, ):
        self.space = self.search_space()
        if self.trials is None:
            self.trials = Trials()

        algo = partial(tpe.suggest, n_startup_jobs=self.n_startup_jobs)
        self.best = fmin(self.classifier_f, self.space, algo=algo, max_evals=30, trials=self.trials)
        before_hold = self.trials.best_trial['result']['loss']
        self.best = fmin(self.classifier_f, self.space, algo=algo, max_evals=60, trials=self.trials)
        after_hold = self.trials.best_trial['result']['loss']

        index = 60
        while (before_hold - after_hold) > 1e-04:
            index = index + 30
            self.best = fmin(self.classifier_f, self.space, algo=algo, max_evals=index, trials=self.trials)
            before_hold, after_hold = after_hold, self.trials.best_trial['result']['loss']

        self.best_loss_result = self.trials.best_trial['result']['loss']
        return self.transform_space()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd
    import time

    random_state = 0

    # data = load_iris()
    # target = data.target
    # data = data.data

    data = pd.read_csv('/r2/data/creditcard_01.csv')
    target = data['Class']
    data.drop('Class', axis=1, inplace=True)

    p = ClassifierParameterOptimization(data, target)
    p.function_min()
    print(p.best_space)
    print(p.estimator.estimator)
    print(p.classifier_f(p.best_space))

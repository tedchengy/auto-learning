from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
from sklearn import metrics


class IncrementalModelBase:
    def __init__(self, ):
        pass

    def fit(self, X, y, **fit_params):
        pass

    def predict(self, X, batch_size=None):
        pass


class IncrementalClassifierModel(IncrementalModelBase):
    def __init__(self, loss='hinge', penalty='l2', l1_ratio=0.15, epsilon=0.1):
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.estimator = None
        self.classes = None
        self.random_state = 0
        self.sample_weight = 5000
        self.sample_index = 3

    def _fit(self, X, y, **kwargs):
        from sklearn.linear_model import SGDClassifier

        estimator = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio,
            epsilon=self.epsilon,
            random_state=self.random_state,
        )

        if self.classes is None:
            classes = y.drop_duplicates()

        for i in range(self.sample_index):
            for xpi, ypi in self.sampling(X, y):
                estimator.partial_fit(xpi, ypi, classes=classes)

        self.estimator = estimator
        return self

    def predict(self, X, **kwargs):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X, )

    def sampling(self, X, y):
        s = self.sample_weight
        loop = int(X.shape[0] / s) + 1
        for i in range(loop):
            yield X[i * s:(i + 1) * s], y[i * s:(i + 1) * s]

    def score(self, X, label):
        y = self.estimator.predict(X)
        if isinstance(y, pd.Series):
            num1 = len(y.drop_duplicates())
        else:
            num1 = len(set(y))

        if isinstance(label, pd.Series):
            num2 = len(label.drop_duplicates())
        else:
            num2 = len(set(label))

        if num1 == 2 and num2 == 2:
            return metrics.roc_auc_score(label, y)
        else:
            return metrics.accuracy_score(label, y)

    def f(self):
        global best_score, count
        # print('####' + str(params))
        count += 1

        try:
            clf = IncrementalClassifierModel(**params)._fit(data, target)
            label = clf.predict(data)
            score = metrics.roc_auc_score(label, target)
        except:
            score = 0

        if score > best_score:
            # print('new best:', score, 'using:', str(params))
            best_score = score
        if count % 100 == 0:
            print('iters:', count, ', score:', score, 'using', params)
        return {'loss': -score, 'status': STATUS_OK}


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

    time_a1 = time.time()
    model_a = SGDClassifier(random_state=random_state)
    model_a.fit(data, target)
    label_a = model_a.predict(data)
    time_a2 = time.time()
    print(metrics.accuracy_score(target, label_a))
    print(metrics.roc_auc_score(target, label_a))
    print(round((time_a2 - time_a1), 5))

    time_b1 = time.time()
    model_b = IncrementalClassifierModel()
    model_b._fit(data, target)
    label_b = model_b.predict(data)
    time_b2 = time.time()
    print(metrics.accuracy_score(target, label_b))
    print(metrics.roc_auc_score(target, label_b))
    print(round((time_b2 - time_b1), 5))

space = {
    'loss': hp.choice('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                               'epsilon_insensitive', 'squared_epsilon_insensitive']),
    'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'epsilon': hp.uniform('epsilon', 0, 1),
}

count = 0
best_score = 0


def f(params):
    global best_score, count
    # print('####', params)
    count += 1

    try:
        clf = IncrementalClassifierModel(**params)._fit(data, target)
        label = clf.predict(data)
        score = metrics.roc_auc_score(label, target)
    except:
        score = 0

    if score > best_score:
        # print(score)
        # print('new best:', score, 'using:', params)
        best_score = score
    if count % 100 == 0:
        print('iters:', count, 'score:', score, 'using', params)
    return {'loss': -score, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=10, trials=trials)
print('best:')
print(best)

'''
class EnsembleSelection(AbstractEnsemble):
    def __init__(
            self,
            ensemble_size: int,
            task_type: int,
            metric: Scorer,
            sorted_initialization: bool = False,
            bagging: bool = False,
            mode: str = 'fast',
            random_state: np.random.RandomState = None,
    ):
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.mode = mode
        self.random_state = random_state

    def fit(self, predictions, labels, identifiers):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not self.task_type in TASK_TYPES:
            raise ValueError('Unknown task type %s.' % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError('Metric must be of type scorer')
        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        if self.bagging:
            self._bagging(predictions, labels)
        else:
            self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(self, predictions, labels):
        if self.mode == 'fast':
            self._fast(predictions, labels)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(self, predictions, labels):
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization:
            n_best = 20
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = calculate_score(
                    labels, ensemble_, self.task_type, self.metric,
                    ensemble_.shape[1])
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for pred in ensemble:
                    ensemble_prediction += pred
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                # TODO: this could potentially be vectorized! - let's profile
                # the script first!
                fant_ensemble_prediction[:, :] = weighted_ensemble_prediction + \
                                                 (1. / float(s + 1)) * pred
                scores[j] = 1 - calculate_score(
                    solution=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]

    def _slow(self, predictions, labels):
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization:
            n_best = 20
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = calculate_score(
                    solution=labels,
                    prediction=ensemble_,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = 1 - calculate_score(
                    solution=labels,
                    prediction=ensemble_prediction,
                    task_type=self.task_type,
                    metric=self.metric,
                    all_scoring_functions=False)
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(order)
        self.trajectory_ = np.array(trajectory)
        self.train_score_ = trajectory[-1]

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _sorted_initialization(self, predictions, labels, n_best):
        perf = np.zeros([predictions.shape[0]])

        for idx, prediction in enumerate(predictions):
            perf[idx] = calculate_score(labels, prediction, self.task_type,
                                        self.metric, predictions.shape[1])

        indices = np.argsort(perf)[perf.shape[0] - n_best:]
        return indices

    def _bagging(self, predictions, labels, fraction=0.5, n_bags=20):
        """Rich Caruana's ensemble selection method with bagging."""
        raise ValueError('Bagging might not work with class-based interface!')
        n_models = predictions.shape[0]
        bag_size = int(n_models * fraction)

        order_of_each_bag = []
        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(random.sample(range(0, n_models), bag_size))
            bag = predictions[indices, :, :]
            order, _ = self._fit(bag, labels)
            order_of_each_bag.append(order)

        return np.array(order_of_each_bag)

    def predict(self, predictions):
        predictions = np.asarray(predictions)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")

    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                          for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            model = models[identifier]
            if weight > 0.0:
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_validation_performance(self):
        return self.trajectory_[-1]
'''

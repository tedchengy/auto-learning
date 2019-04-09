# auto-learning
## auto-lncremental-learning
#### 功能：自动增量学习
## intro 
#### 简介
auto-learning is a python library that wraps sklearn partial_fit and hyperopt for auto lncremental learning.It runs much fast.
## sample
#### 示例
```python
from autolearning.ensembles.parameter_optimization import HyperoptClassifierParameter
from sklearn.datasets import load_iris


data = load_iris()
target = data.target
data = data.data


p = ClassifierParameterOptimization(data, target)
p.function_min()
print(p.best_space)
print(p.estimator.estimator)
print(p.classifier_f(p.best_space))


    from sklearn.datasets import load_iris
    from sklearn.datasets import load_boston
    import pandas as pd

    random_state = 0

    data = load_boston()
    # data = load_iris()
    target = data.target
    data = data.data

    # data = pd.read_csv('/r2/data/creditcard_01.csv')
    # target = data['Class']
    # data.drop('Class', axis=1, inplace=True)

    # p = ClassifierParameterOptimization(data, target)
    p = RegressorParameterOptimization(data, target)
    p.function_min()
    print(p.best_space)
    print(p.estimator.estimator)
    print(p.function(p.best_space))

```


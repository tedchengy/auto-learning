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


p = HyperoptClassifierParameter(data, target)
p.function_min()
print(p.best_space)
print(p.estimator.estimator)
print(p.classifier_f(p.best_space))
```


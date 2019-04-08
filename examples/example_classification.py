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
from autolearning.ensembles.parameter_optimization import ClassifierParameterOptimization
from sklearn.datasets import load_iris


data = load_iris()
target = data.target
data = data.data


p = ClassifierParameterOptimization(data, target)
p.function_min()
print(p.best_space)
print(p.estimator.estimator)
print(p.function(p.best_space))

\
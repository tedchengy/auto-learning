from autolearning.ensembles.parameter_optimization import RegressorParameterOptimization
from sklearn.datasets import load_boston


data = load_boston()
target = data.target
data = data.data


p = RegressorParameterOptimization(data, target)
p.function_min()
print(p.best_space)
print(p.estimator.estimator)
print(p.function(p.best_space))

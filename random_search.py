import time

import numpy as np
import pandas as pd

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import tabulate

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def sequential_random_search(model, configspace, budget, X, y, cv=False):
    data = []
    for b in range(budget):
        # Get a configuration to try
        config = cs.sample_configuration().get_dictionary()
        clf = model(**config)
        if not cv:
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X,y)
            clf.fit(X_train, y_train)
            config['score'] = clf.score(X_val, y_val)
        else:
            config['score'] = np.mean(cross_val_score(clf, X, y, cv=5))
        data.append(config)
    return pd.DataFrame(data)


# Load the data: Australian dataset
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
)

# Define the model to use
model = GradientBoostingClassifier
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

# Define the configuration space
cs = CS.ConfigurationSpace()
# learning rate shrinks the contribution of each tree by learning_rate
learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.2, default_value=0.1, log=True)
cs.add_hyperparameter(learning_rate)
# The number of boosting stages to perform
n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=10, upper=100, default_value=100)
cs.add_hyperparameter(n_estimators)
# The fraction of samples to be used for fitting the individual base learners.
subsample = CSH.UniformFloatHyperparameter('subsample', lower=0.5, upper=1.0, default_value=1.0)
cs.add_hyperparameter(subsample)
# The function to measure the quality of a split
criterion = CSH.CategoricalHyperparameter('criterion', ['friedman_mse', 'mae', 'mse'])
cs.add_hyperparameter(criterion)
# The minimum number of samples required to split an internal node
min_samples_split = CSH.UniformIntegerHyperparameter('min_samples_split', lower=1, upper=12, default_value=2)
cs.add_hyperparameter(min_samples_split)
# The minimum number of samples required to be at a leaf node
min_samples_leaf = CSH.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=12, default_value=2)
cs.add_hyperparameter(min_samples_leaf)
# The maximum depth of the individual regression estimators.
max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=3, upper=8, default_value=3)
cs.add_hyperparameter(max_depth)
# The number of features to consider when looking for the best split
max_features = CSH.CategoricalHyperparameter('max_features', ['log2', 'sqrt'])
cs.add_hyperparameter(max_features)

budget = 2
start_time = time.time()
results_frame = sequential_random_search(model=model, configspace=cs, budget=budget, X=X_train, y=y_train, cv=False)
print(f"Elapsed time for sequential search = {time.time() - start_time}")

print(tabulate.tabulate(results_frame, headers='keys', tablefmt='psql'))

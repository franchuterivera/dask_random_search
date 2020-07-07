import collections

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from sklearn.ensemble import GradientBoostingClassifier


import numpy as np


# Create a run object
RunInfo = collections.namedtuple('RunInfo', 'run_id model dataset configuration cv')


class Dispatcher:
    def __init__(self, budget, cv=True):
        self.original_budget = budget
        self.cv = cv

        self.dispatched_jobs = 0
        self.finished_jobs = 0

        # Get the configuration space
        self.cs = self._get_config_space()

        # Model to run
        self.model = GradientBoostingClassifier

        self.dataset = 40981

    def get_next_run(self):
        run_info = RunInfo(
            run_id=self.dispatched_jobs,
            model=self.model,
            dataset=self.dataset,
            configuration=self.cs.sample_configuration(),
            cv=self.cv,
        )
        self.dispatched_jobs += 1
        return run_info

    def is_complete(self):
        return self.dispatched_jobs >= self.original_budget

    def _get_config_space(self):
        # Define the configuration space
        cs = CS.ConfigurationSpace()
        # learning rate shrinks the contribution of each tree by learning_rate
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.2,           default_value=0.1, log=True)
        cs.add_hyperparameter(learning_rate)
        # The number of boosting stages to perform
        n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=10, upper=100,             default_value=100)
        cs.add_hyperparameter(n_estimators)
        # The fraction of samples to be used for fitting the individual base learners.
        subsample = CSH.UniformFloatHyperparameter('subsample', lower=0.5, upper=1.0, default_value=1.0)
        cs.add_hyperparameter(subsample)
        # The function to measure the quality of a split
        criterion = CSH.CategoricalHyperparameter('criterion', ['friedman_mse', 'mae', 'mse'])
        cs.add_hyperparameter(criterion)
        # The minimum number of samples required to split an internal node
        min_samples_split = CSH.UniformIntegerHyperparameter('min_samples_split', lower=2, upper=12,     default_value=2)
        cs.add_hyperparameter(min_samples_split)
        # The minimum number of samples required to be at a leaf node
        min_samples_leaf = CSH.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=12,       default_value=2)
        cs.add_hyperparameter(min_samples_leaf)
        # The maximum depth of the individual regression estimators.
        max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=3, upper=8, default_value=3)
        cs.add_hyperparameter(max_depth)
        # The number of features to consider when looking for the best split
        max_features = CSH.CategoricalHyperparameter('max_features', ['log2', 'sqrt'])
        cs.add_hyperparameter(max_features)
        return cs



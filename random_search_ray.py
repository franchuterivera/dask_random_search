import collections
import time

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 999)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

import ray

# Create a run object
RunInfo = collections.namedtuple('RunInfo', 'run_id configuration')


# Function to map reduce
@ray.remote(num_cpus=1)
def fit_and_score(run_info):
    start_time = time.time()
    time.sleep(np.random.randint(10))
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True)
    clf = GradientBoostingClassifier(**run_info.configuration.get_dictionary())
    score = np.mean(cross_val_score(clf, X, y, cv=5))
    duration = time.time() - start_time

    return_dict = {
        'score': score,
        'duration': duration,
    }
    return_dict.update(run_info.configuration.get_dictionary())
    return return_dict


# Define the configuration space
cs = CS.ConfigurationSpace()
# learning rate shrinks the contribution of each tree by learning_rate
learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.2,
                                               default_value=0.1, log=True)
cs.add_hyperparameter(learning_rate)
# The number of boosting stages to perform
n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=10, upper=100,
                                                default_value=100)
cs.add_hyperparameter(n_estimators)
# The fraction of samples to be used for fitting the individual base learners.
subsample = CSH.UniformFloatHyperparameter('subsample', lower=0.5, upper=1.0, default_value=1.0)
cs.add_hyperparameter(subsample)
# The function to measure the quality of a split
criterion = CSH.CategoricalHyperparameter('criterion', ['friedman_mse', 'mae', 'mse'])
cs.add_hyperparameter(criterion)
# The minimum number of samples required to split an internal node
min_samples_split = CSH.UniformIntegerHyperparameter('min_samples_split', lower=2, upper=12,
                                                     default_value=2)
cs.add_hyperparameter(min_samples_split)
# The minimum number of samples required to be at a leaf node
min_samples_leaf = CSH.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=12,
                                                    default_value=2)
cs.add_hyperparameter(min_samples_leaf)
# The maximum depth of the individual regression estimators.
max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=3, upper=8, default_value=3)
cs.add_hyperparameter(max_depth)
# The number of features to consider when looking for the best split
max_features = CSH.CategoricalHyperparameter('max_features', ['log2', 'sqrt'])
cs.add_hyperparameter(max_features)


class OptimizationAlgorithm:
    def __init__(self, cs, budget):
        self.original_budget = budget
        self.cs = cs

        self.dispatched_jobs = 0
        self.finished_jobs = 0

    def get_next_run(self):
        run_info = RunInfo(
            run_id=self.dispatched_jobs,
            configuration=self.cs.sample_configuration(),
        )
        self.dispatched_jobs += 1
        return run_info

    def report_result(self, result):
        pass

    def is_complete(self):
        return self.dispatched_jobs >= self.original_budget

if __name__== "__main__" :

    # Create a job dispatcher
    dispatcher = OptimizationAlgorithm(cs, budget=25)

    # Setup the cluster
    ray.init(num_cpus=4)

    run_history = []
    obj_refs = []

    # Submit all jobs
    counter = 0

    while not dispatcher.is_complete():

        obj_refs_completed = []

        # Are there workers available?
        # The nthreads dynamically query for the scheduler capacity
        # On a local cluster, capacity can be increased by scale()
        # On a distributed, dask-worker 'tcp://127.0.0.1:45739' --nprocs 1 --nthreads 1
        total_compute_power = ray.available_resources()['CPU']
        if len(obj_refs) < total_compute_power:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            print(f"Scheduling job {counter} at {now}")
            counter += 1

            # submit a new job
            run_info = dispatcher.get_next_run()
            obj_refs.append(
                fit_and_score.remote(run_info)
            )

            # I am not able to find a nice way to get all completed jobs,
            # ray.objects() doesn't return anything, and ray.available resources
            # is not reliable because it takes a little bit of time to update the
            # database

        else:
            # If the cluster is under full load, we simply wait until any object is completed
            # then move on to the next iteration
            obj_refs_completed, obj_refs = ray.wait(obj_refs, num_returns=1, timeout=None)

        for obj_ref in obj_refs_completed:
            run_history.append(ray.get(obj_ref))

    # Wait for any remaining futures after all
    # runs have been dispatched
    for result in ray.get(obj_refs):
        run_history.append(result)

    print(pd.DataFrame(run_history))
    ray.shutdown()

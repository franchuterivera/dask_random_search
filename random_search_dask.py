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

from dask.distributed import Client, wait


# Create a run object
RunInfo = collections.namedtuple('RunInfo', 'run_id configuration')


# Function to map reduce
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


def are_workers_available(client):

    # Scheduler info returns a dict with:
    # type, id, address, services, workers
    # each worker has:
    #   type, id, host, resource, name, nthreads, memory_limit, metrics, executing,
    #   in_memory, ready, bandwith
    workers = client.scheduler_info()['workers']
    executing = sum(v['metrics']['executing'] for v in workers.values())
    return 0 if executing >= total_compute_power else total_compute_power - executing

if __name__== "__main__" :

    # Create a job dispatcher
    dispatcher = OptimizationAlgorithm(cs, budget=25)

    # Setup the cluster
    client = Client(n_workers=4, processes=True, threads_per_worker=1)

    # For the sake of documentation,
    # one could also know if there are workers by doing
    # are_workers_available. It takes a couple of seconds
    # for the executing variable to be populated, so that
    # is dangerous, for instance when launching all runs,
    # which takes miliseconds, so everythign is run
    # Also, # wait_for_workers(n_workers=0)
    # just wait for a valid worker, not an empty worker
    # The safest approach is to track futures!

    run_history = []
    futures = []

    # Submit all jobs
    counter = 0

    while not dispatcher.is_complete():

        # Are there workers available?
        # The nthreads dynamically query for the scheduler capacity
        # On a local cluster, capacity can be increased by scale()
        # On a distributed, dask-worker 'tcp://127.0.0.1:45739' --nprocs 1 --nthreads 1
        total_compute_power = sum(client.nthreads().values())
        if len(futures) < total_compute_power:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            print(f"Scheduling job {counter} at {now}")
            counter += 1

            # submit a new job
            run_info = dispatcher.get_next_run()
            futures.append(
                client.submit(fit_and_score, run_info)
            )

            # since the last iteration and immediately check if we need to submit more jobs
            done_futures = [f for f in futures if f.done()]

        else:
            # If the cluster is under full load, we simply wait until any future is completed and
            # then move on to the next iteration
            done_futures = wait(futures, return_when='FIRST_COMPLETED').done

        for future in done_futures:
            run_history.append(future.result())
            futures.remove(future)

    # Wait for any remaining futures after all
    # runs have been dispatched
    for result in client.gather(futures):
        run_history.append(result)

    print(pd.DataFrame(run_history))
    client.shutdown()

import time

import numpy as np
import pandas as pd

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import tabulate

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import cross_val_score

from dask.distributed import Client

from job_dispatcher import Dispatcher

# Function to map reduce
def fit_and_score(run_info):
    start_time = time.time()
    time.sleep(np.random.randint(10))
    # Load the data: Australian dataset 40981
    X, y = sklearn.datasets.fetch_openml(data_id=run_info.dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
    )
    clf = run_info.model(**run_info.configuration.get_dictionary())
    if not run_info.cv:
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X,y)
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
    else:
        score = np.mean(cross_val_score(clf, X, y, cv=5))
    duration = time.time() - start_time

    return_dict = {
        'score' : score,
        'duration' : duration,
    }
    return_dict.update(run_info.configuration.get_dictionary())
    return return_dict


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
    dispatcher = Dispatcher(budget=25)

    # Setup the cluster
    client = Client(n_workers=4)
    client.cluster
    workers = client.scheduler_info()['workers']
    total_compute_power = sum(v['nthreads'] for v in workers.values())

    # For the sake of documentation.
    # one could also know if there are workers by doing
    # are_workers_available. It takes a couple of seconds
    # for the executing variable to be populated, so that
    # is dangerous, for instance when launching all runs,
    # which takes miliseconds, so everythign is run
    # Also,
    # wait_for_workers(n_workers=0)
    # just wait for a valid worker, not an empty worker
    # The safest approach is to track futures!

    run_history = []
    futures = []

    # Submit all jobs
    counter = 0
    while not dispatcher.is_complete():

        # Are there workers available?
        if len(futures) < total_compute_power:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            print(f"Scheduling job {counter} at {now}")
            counter += 1

            # submit a new job
            run_info = dispatcher.get_next_run()
            futures.append(
                client.submit(fit_and_score, run_info)
            )

        # Check if a future finished and register
        # it to the run_history
        finished_runs = [i for i in range(len(futures)) if futures[i].done()]

        # If there is a finished run, register it and
        # forget about the future
        for i in sorted(finished_runs, reverse=True):
            # get the RunInfo Object
            result = futures[i].result()

            # Add to run history
            run_history.append(result)
            del futures[i]

    # Wait for any remaining futures after all
    # runs have been dispatched
    for result in client.gather(futures):
        run_history.append(result)

    print(tabulate.tabulate(
        pd.DataFrame(run_history),
        headers='keys',
        tablefmt='psql'
    ))
    client.shutdown()

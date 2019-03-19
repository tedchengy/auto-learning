# -*- encoding: utf-8 -*-
"""
====================
Parallel Usage
====================

*Auto-sklearn* uses *SMAC* to automatically optimize the hyperparameters of
the training models. A variant of *SMAC*, called *pSMAC* (parallel SMAC),
provides a means of running several instances of *auto-sklearn* in a parallel
mode using several computational resources (detailed information of
*pSMAC* can be found `here <https://automl.github.io/SMAC3/stable/psmac.html>`_).
This example shows the necessary steps to configure *auto-sklearn* in
parallel mode.
"""

import multiprocessing
import shutil

import sklearn.model_selection
import sklearn.datasets
from sklearn import metrics
from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *

import pandas as pd
from sklearn.model_selection import train_test_split

tmp_folder = '/tmp/autosklearn_parallel_example_tmp'
output_folder = '/tmp/autosklearn_parallel_example_out'

MINUTE = 5
MEMORY = 8
PER = 1

for dir in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        pass


def get_spawn_classifier(X_train, y_train):
    def spawn_classifier(seed, dataset_name):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = 25
            smac_scenario_args = {}
        else:
            initial_configurations_via_metalearning = 0
            smac_scenario_args = {'initial_incumbent': 'RANDOM'}

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60 * MINUTE,  # sec., how long should this seed fit process run
            per_run_time_limit=60 * PER,  # sec., each model may only take this long before it's killed
            ml_memory_limit=1024 * MEMORY,  # MB, memory limit imposed on each call to a ML algorithm
            shared_mode=True,  # tmp folder will be shared between seeds
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=False,
            ensemble_size=0,  # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            seed=seed,
            smac_scenario_args=smac_scenario_args,
        )
        automl.fit(X_train, y_train, dataset_name=dataset_name)

    return spawn_classifier


if __name__ == '__main__':

    data = pd.read_csv("~/data/creditcard.csv")
    targets = pd.DataFrame()
    targets = data['Class']
    data.drop('Class', axis=1, inplace=True)

    '''
    data = pd.read_csv("titanic.train.csv")
    targets = data['survived']
    data['Gender']=data['sex'].map({'female':0,'male':1}).astype(int)
    data.drop(['survived','name','ticket','embarked','cabin','sex'],axis=1, inplace=True)
    '''

    '''
    data = pd.read_csv("500wX50_classification_data_output.csv")
    targets = data['Y']
    data.drop(['Y'],axis=1, inplace=True)
    '''

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.30, random_state=1)

    '''
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    '''

    processes = []
    spawn_classifier = get_spawn_classifier(X_train, y_train)
    for i in range(4):  # set this at roughly half of your cores
        p = multiprocessing.Process(
            target=spawn_classifier,
            args=(i, 'breast_cancer'),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print('Starting to build an ensemble!')
    automl = AutoSklearnClassifier(
        time_left_for_this_task=60 * MINUTE,
        per_run_time_limit=60 * PER,
        ml_memory_limit=1024 * MEMORY,
        shared_mode=True,
        ensemble_size=50,
        ensemble_nbest=200,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning=0,
        seed=1,
    )

    # Both the ensemble_size and ensemble_nbest parameters can be changed now if
    # necessary
    automl.fit_ensemble(
        y_train,
        task=MULTICLASS_CLASSIFICATION,
        metric=accuracy,
        precision='32',
        dataset_name='digits',
        ensemble_size=20,
        ensemble_nbest=50,
    )

    predictions = automl.predict(X_test)
    print(automl.show_models())
    print("Accuracy score", metrics.accuracy_score(y_test, predictions))
    print("roc_auc_score:\n ", metrics.roc_auc_score(y_test, predictions))
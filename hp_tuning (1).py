from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
import hyperopt
import numpy as np
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
import optuna
import scipy
from hyperopt.pyll.stochastic import sample
from time import ctime
from math import log, ceil
from random import random


def roc_auc_scorer():

    def roc_auc(y_true, pred):
        return roc_auc_score(y_true=y_true,
                             y_score=pred)

    return make_scorer(roc_auc,
                needs_proba=True,
                greater_is_better=True)


class HyperoptObjective(object):
    def __init__(self,
                 train_pool,
                 val_pool,
                 model,
                 const_params,
                 fit_params,
                 cv_splitter,
                 cv_scoring,
                 cat_features=None):
        self.evaluated_count = 0
        self.train_pool = train_pool
        self.val_pool = val_pool
        self.model = model
        self.constant_params = const_params
        self.fit_params = fit_params
        self.cv_splitter = cv_splitter
        self.cv_scoring = cv_scoring
        self.cat_features = cat_features
        #self.X_val, self.y_val = val_pool.get_features(), val_pool.get_label()

    '''
    The way that HyperOpt fmin function works, is that on each evaluation 
    it calls given objective function. 
    '''

    def __call__(self,
                 hyper_params):
        names = ['learning_rate', 'random_strength', 'l2_leaf_reg', 'subsample', 'leaf_estimation_iterations']
        if type(hyper_params) == list:
            hp = {}
            for i in range(5):
                hp[names[i]] = hyper_params[i]
            model = self.model(**hp, **self.constant_params)
        else:
            model = self.model(**hyper_params, **self.constant_params)
        model.fit(self.train_pool,
                  #cat_features=self.cat_features,
                  eval_set=self.val_pool,
                  **self.fit_params)
                  
        #y_pred = model.predict_proba(self.X_val)

        mean_cv_result = model.eval_metrics(self.val_pool,
                                            metrics=['AUC'],
                                            ntree_start=(model.tree_count_ - 1))['AUC'][0]
        print('------------------ RES:', mean_cv_result)

        self.evaluated_count += 1

        '''
        Hyperopt always tries to minimize loss. We will check _sign parametter attached 
        to scorer function by make_scorer (it is inferred from greater_is_better parameter). 
        If _sign is positive, we will flip mean_cv_result sign to convert score to loss. 
        '''
        if self.cv_scoring._sign > 0:
            mean_cv_result = -mean_cv_result

        return mean_cv_result


class GaussianObjective(HyperoptObjective):
    def __call__(self,
                 hyper_params):
        names = ['learning_rate', 'random_strength', 'l2_leaf_reg', 'subsample', 'leaf_estimation_iterations']
        hp = {}
        for i in range(5):
            hp[names[i]] = hyper_params[i]
        print(hp)
        model = self.model(**hp, **self.constant_params)
        model.fit(self.train_pool,
                  #cat_features=self.cat_features,
                  eval_set=self.val_pool,
                  **self.fit_params)
        #y_pred = model.predict_proba(self.X_val)

        #mean_cv_result = roc_auc_score(self.y_val, y_pred[:, 1])
        mean_cv_result = model.eval_metrics(self.val_pool,
                                            metrics=['AUC'],
                                            ntree_start=(model.tree_count_ - 1))['AUC'][0]
        print('------------------ RES:', mean_cv_result)

        self.evaluated_count += 1

        return -mean_cv_result


class OptunaObjective(HyperoptObjective):
    def __call__(self,
                 trial):
        hp = {
            "subsample": trial.suggest_uniform("subsample", 0., 1.),
            'random_strength': trial.suggest_int('random_strength', 1, 21),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 11),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.999999),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1., 10.)
        }
        model = self.model(**hp, **self.constant_params)
        model.fit(self.train_pool,
                  #cat_features=self.cat_features,
                  eval_set=self.val_pool,
                  **self.fit_params)
        #y_pred = model.predict_proba(self.X_val)

        #mean_cv_result = roc_auc_score(self.y_val, y_pred[:, 1])
        mean_cv_result = model.eval_metrics(self.val_pool,
                                            metrics=['AUC'],
                                            ntree_start=(model.tree_count_ - 1))['AUC'][0]
        print('------------------ RES:', mean_cv_result)

        self.evaluated_count += 1

        return mean_cv_result


def find_best_params(train_pool,
                     val_pool,
                     model,
                     const_params,
                     parameter_space,
                     fit_params=None,
                     max_evals=25,
                     cv_splitter=None,
                     cv_scoring=None,
                     cat_features=None):

    objective = HyperoptObjective(train_pool,
                                  val_pool,
                                  model,
                                  const_params,
                                  fit_params,
                                  cv_splitter,
                                  cv_scoring,
                                  cat_features)
    '''
    HyperOpt Trials object stores details of every iteration. 
    '''
    trials = hyperopt.Trials()

    best_params = hyperopt.fmin(fn=objective,
                                space=parameter_space,
                                algo=hyperopt.tpe.suggest,
                                rstate=np.random.RandomState(seed=42),
                                max_evals=max_evals,
                                trials=trials)

    best_params.update(const_params)

    return best_params, trials


def print_results(train_pool,
                  val_pool,
                  test_pool,
                  model):
    #X_train, y_train = train_pool.get_features(), train_pool.get_label()
    #X_val, y_val = val_pool.get_features(), val_pool.get_label()
    #X_test, y_test = test_pool.get_features(), test_pool.get_label()
    results = {}
    results = {}
    for name, pool in zip(['Train', 'Val', 'Test'],
                          [train_pool, val_pool, test_pool]):
        #y_predicted = model.predict_proba(pool)
        acc_res = model.score(pool)
        roc_auc_res = model.eval_metrics(pool, metrics=['AUC'], ntree_start=(model.tree_count_ - 1))['AUC'][0]
        #roc_auc_score(y, y_predicted[:, 1])
        print("------ %s Accuracy = %f" % (name, acc_res))
        print("------ %s ROC AUC = %f\n" % (name, roc_auc_res))
        results[name + "_acc"] = acc_res
        results[name + "_roc_auc"] = roc_auc_res
    return results


def catboost_from_cfg(train_pool,
                      val_pool,
                      test_pool,
                      cat_features,
                      n_estimators=1000,
                      max_evals=25,
                      task_type="CPU"):
    """
    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!
    Returns:
    --------
    A mean score for the classifier on the loaded data-set.
    """
    X_val, y_val = val_pool.get_features(), val_pool.get_label()

    def func(cfg):
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        cfg['task_type'] = task_type
        cfg['n_estimators'] = n_estimators
        clf = CatBoostClassifier(**cfg)
        clf.fit(train_pool,
                cat_features=cat_features,
                eval_set=val_pool,
                logging_level='Silent')
        #y_pred = clf.predict_proba(X_val)

        #mean_cv_result = roc_auc_score(y_val, y_pred[:, 1])
        mean_cv_result = model.eval_metrics(val_pool,
                                            metrics=['AUC'],
                                            ntree_start=(model.tree_count_ - 1))['AUC'][0]
        print('------------------ RES:', mean_cv_result)
        return -mean_cv_result
    return func  # Minimize!


class Hyperband:

    def __init__(self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 81  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                print("\n*** {} configurations x {:.1f} iterations each".format(n_configs, n_iterations))

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter))

                    start_time = time.time()

                    #if dry_run:
                    #    result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    #else:
                    result = self.try_params(n_iterations, t)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time.time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results


def train_and_eval_catboost_classifier(clf,
                                       train_pool,
                                       val_pool,
                                       test_pool,
                                       cat_features):

    clf.fit(train_pool,
            eval_set=val_pool,
            logging_level='Silent')


    res = {}

    acc_res = clf.score(train_pool)
    roc_auc_res = clf.eval_metrics(train_pool, metrics=['AUC'], ntree_start=(clf.tree_count_ - 1))['AUC'][0]

    print("\n# training | AUC: {:.2%}, accuracy: {:.2%}".format(roc_auc_res, acc_res))

    res['Train_acc'] = acc_res
    res['Train_roc_auc'] = roc_auc_res

    acc_res = clf.score(test_pool)
    roc_auc_res = clf.eval_metrics(test_pool, metrics=['AUC'], ntree_start=(clf.tree_count_ - 1))['AUC'][0]

    res['Test_acc'] = acc_res
    res['Test_roc_auc'] = roc_auc_res

    print("# testing  | AUC: {:.2%}, accuracy: {:.2%}".format(roc_auc_res, acc_res))

    acc_res = clf.score(val_pool)
    roc_auc_res = clf.eval_metrics(val_pool, metrics=['AUC'], ntree_start=(clf.tree_count_ - 1))['AUC'][0]

    res['Val_acc'] = acc_res
    res['Val_roc_auc'] = roc_auc_res

    print("# validation  | AUC: {:.2%}, accuracy: {:.2%}".format(roc_auc_res, acc_res))

    res['loss'] = -roc_auc_res
    res['auc'] = roc_auc_res
    res['n_estimators'] = clf.tree_count_

    # return { 'loss': 1 - auc, 'log_loss': ll, 'auc': auc }
    # return { 'loss': ll, 'log_loss': ll, 'auc': auc }
    return res


def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params


def get_params():
    space = {
        'learning_rate': hyperopt.hp.choice('lr', [hyperopt.hp.loguniform('lr_', -5., 0.)]),
        'subsample': hyperopt.hp.choice('ss', [hyperopt.hp.uniform('ss_', 0., 1.)]),
        'l2_leaf_reg': hyperopt.hp.choice('l2lr', [hyperopt.hp.loguniform('l2lr_', 0., np.log(10.))]),
        'random_strength': hyperopt.hp.choice('rs', [hyperopt.hp.choice('rs_', np.arange(1, 21))]),
        'leaf_estimation_iterations': hyperopt.hp.choice('lei', [hyperopt.hp.choice('lei_', np.arange(1, 11))])
    }

    params = sample(space)
    params = {k: v for k, v in params.items() if v is not 'default'}
    return handle_integers(params)



def try_params(train_pool, val_pool, test_pool, cat_features, n_estimators, task_type='CPU', bootstrap_type='MVS'):
    def func(n_iterations, params):
        trees_per_iteration = max(1, n_estimators // 27)
        n_estimators_ = int(round(n_iterations * trees_per_iteration))
        print("n_estimators:", n_estimators_)
        clf = CatBoostClassifier(n_estimators=n_estimators_,
                                 task_type=task_type,
                                 bootstrap_type=bootstrap_type,
                                 **params)
        return train_and_eval_catboost_classifier(clf, train_pool, val_pool, test_pool, cat_features)
    return func


def choose_classic(train_pool,
                   val_pool,
                   test_pool,
                   cat_features,
                   n_estimators=1000,
                   max_evals=25,
                   task_type="CPU"):

    model = CatBoostClassifier(task_type=task_type, n_estimators=n_estimators)
    model.fit(train_pool,
              #cat_features=cat_features,
              eval_set=val_pool,
              logging_level='Silent')
    print("CatBoost Classic")
    results = print_results(train_pool,
                            val_pool,
                            test_pool,
                            model)
    return results


def choose_hyperOpt(train_pool,
                    val_pool,
                    test_pool,
                    cat_features,
                    n_estimators=1000,
                    max_evals=25,
                    task_type="CPU"):
    const_params = {'n_estimators': n_estimators,
                    'task_type': task_type}
    if task_type == 'CPU':
        const_params['bootstrap_type'] = 'MVS'
    else:
        const_params['bootstrap_type'] = 'Poisson'

    parameter_space = {
        'learning_rate': hyperopt.hp.loguniform('learning_rate', -5., 0.),
        'random_strength': hyperopt.hp.choice('random_strength', np.arange(1, 21)),
        'l2_leaf_reg': hyperopt.hp.loguniform('l2_leaf_reg', 0., np.log(10)),
        'subsample': hyperopt.hp.uniform('subsample', 0., 1.),
        'leaf_estimation_iterations': hyperopt.hp.choice('leaf_estimation_iterations', np.arange(1, 11))
    }

    fit_params = {
        'logging_level': 'Silent'
    }
    best_params, trials = find_best_params(train_pool,
                                           val_pool,
                                           CatBoostClassifier,
                                           const_params,
                                           parameter_space,
                                           fit_params,
                                           max_evals=max_evals,
                                           cv_splitter=StratifiedKFold(n_splits=2),
                                           cv_scoring=roc_auc_scorer(),
                                           cat_features=cat_features)
    best_params['n_estimators'] = n_estimators
    best_params['bootstrap_type'] = const_params['bootstrap_type']
    model = CatBoostClassifier(**best_params)
    model.fit(train_pool,
              #cat_features=cat_features,
              eval_set=val_pool,
              logging_level='Silent')
    print("CatBoost + HyperOpt")
    results = print_results(train_pool,
                  val_pool,
                  test_pool,
                  model)
    return results


def choose_gaussian(train_pool,
                    val_pool,
                    test_pool,
                    cat_features,
                    n_estimators=1000,
                    max_evals=25,
                    task_type="CPU"):
    const_params = {'n_estimators': n_estimators,
                    'task_type': task_type}
    if task_type == 'CPU':
        const_params['bootstrap_type'] = 'MVS'
    else:
        const_params['bootstrap_type'] = 'Poisson'
    fit_params = {
        'logging_level': 'Silent'
    }
    cv_splitter = StratifiedKFold(n_splits=2)
    cv_scoring = roc_auc_scorer
    names = ['learning_rate', 'random_strength', 'l2_leaf_reg', 'subsample', 'leaf_estimation_iterations']
    space = [Real(0.0001, 5., prior='log-uniform', name=names[0]),
             Integer(1, 21, name=names[1]),
             Real(0.0001, np.log(10), prior='log-uniform', name=names[2]),
             Real(0.1, 0.9999, prior='uniform', name=names[3]),
             Integer(1, 11, name=names[4])]
    objective = GaussianObjective(train_pool,
                                  val_pool,
                                  CatBoostClassifier,
                                  const_params,
                                  fit_params,
                                  cv_splitter,
                                  cv_scoring,
                                  cat_features)
    res_gp = gp_minimize(objective,
                         space,
                         n_calls=max_evals,
                         random_state=0)
    best_params = {}
    if type(res_gp.x) == dict:
        for key in res_gp.x.keys():
            best_params[key] = res_gp.x[key]
    else:
        print(res_gp.x)
        for i in range(5):
            best_params[names[i]] = res_gp.x[i]
    best_params['n_estimators'] = n_estimators
    best_params['bootstrap_type'] = const_params['bootstrap_type']
    model = CatBoostClassifier(**best_params)
    model.fit(train_pool,
              #cat_features=cat_features,
              eval_set=val_pool,
              logging_level='Silent')
    print("CatBoost + GaussianProcesses")
    results = print_results(train_pool,
                  val_pool,
                  test_pool,
                  model)
    return results


def choose_optuna(train_pool,
                  val_pool,
                  test_pool,
                  cat_features,
                  n_estimators=1000,
                  max_evals=25,
                  task_type="CPU"):
    const_params = {'n_estimators': n_estimators,
                    'task_type': task_type}
    if task_type == 'CPU':
        const_params['bootstrap_type'] = 'MVS'
    else:
        const_params['bootstrap_type'] = 'Poisson'
    fit_params = {
        'logging_level': 'Silent'
    }
    cv_splitter = StratifiedKFold(n_splits=2)
    cv_scoring = roc_auc_scorer
    study = optuna.create_study(direction="maximize")
    study.optimize(OptunaObjective(train_pool,
                                   val_pool,
                                   CatBoostClassifier,
                                   const_params,
                                   fit_params,
                                   cv_splitter,
                                   cv_scoring,
                                   cat_features),
                   n_trials=max_evals,
                   timeout=600)
    trial = study.best_trial
    trial.params['n_estimators'] = n_estimators
    trial.params['bootstrap_type'] = const_params['bootstrap_type']

    model = CatBoostClassifier(**trial.params)
    model.fit(train_pool,
              #cat_features=cat_features,
              eval_set=val_pool,
              logging_level='Silent')
    print("CatBoost + Optuna")
    results = print_results(train_pool,
                  val_pool,
                  test_pool,
                  model)
    return results


def choose_hyperband2(train_pool,
                      val_pool,
                      test_pool,
                      cat_features,
                      n_estimators=1000,
                      max_evals=25,
                      task_type="CPU"):
    hb = Hyperband(get_params, try_params(train_pool,
                                          val_pool,
                                          test_pool,
                                          cat_features,
                                          n_estimators,
                                          task_type))
    results = hb.run(skip_last=1)
    print("{} total, best:\n".format(len(results)))
    best_results = {}

    for r in sorted(results, key=lambda x: x['loss'])[:1]:
        print("val_auc: {:.4} | {} seconds | {:.1f} iterations | run {} ".format(
            r['auc'] / 100., r['seconds'], r['iterations'], r['counter']))
        for key in ["Train_acc", "Train_roc_auc", "Test_acc", "Test_roc_auc", "Val_acc", "Val_roc_auc"]:
            best_results[key] = r[key]
        print()

    return best_results


def hp_tuning(train, test, validate, column_description, algo_name, result_path):
    train_pool = Pool(train, column_description=column_description)
    val_pool = Pool(validate, column_description=column_description)
    test_pool = Pool(test, column_description=column_description)
    cat_features = train_pool.get_cat_feature_indices()

    n_estimators = 1000
    max_evals = 50
    task_type = 'CPU'
    
    if algo_name == 'classic':
        func = choose_classic
    elif algo_name == 'hyperopt':
        func = choose_hyperOpt
    elif algo_name == 'gaussian':
        func = choose_gaussian
    elif algo_name == 'optuna':
        func = choose_optuna
    elif algo_name == 'hyperband':
        func = choose_hyperband2
    else:
        print("INCORRECT ALGO NAME!!!")
        return
    start = time.time()
    results = func(train_pool,
                   val_pool,
                   test_pool,
                   cat_features,
                   n_estimators=n_estimators,
                   max_evals=max_evals,
                   task_type=task_type)
    end = time.time()
    results["time"] = end - start
    with open(result_path, 'w') as outfile:
        json.dump(results, outfile)


import sys
import json
import time

if __name__ == '__main__':
    hp_tuning(*sys.argv[1:])

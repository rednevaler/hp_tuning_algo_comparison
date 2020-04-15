from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
import hyperopt
import numpy as np
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
import optuna
import scipy
from dask_ml.model_selection import HyperbandSearchCV
from distributed import Client


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

        mean_cv_result = model.eval_metrics(val_pool,
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
        mean_cv_result = model.eval_metrics(val_pool,
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
        mean_cv_result = model.eval_metrics(val_pool,
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


def choose_hyperband(X_train,
                     X_test,
                     X_val,
                     y_train,
                     y_test,
                     y_val,
                     cat_features,
                     n_estimators=1000,
                     max_evals=25,
                     task_type="CPU"):
    client = Client()
    const_params = {'n_estimators': n_estimators,
                    'task_type': task_type}
    for i in cat_features:
        print(i)
        if type(X_train) in [list, np.array, np.ndarray]:
            unique_feature = list(set(X_train[:, i]).add(X_val[:, i]).add(X_test[:, i]))
        else:
            column = X_train.columns[i]
            unique_feature = list(
                    set(list(X_train[column])).union(
                    set(list(X_val[column]))).union(
                    set(list(X_test[column]))))
        unique_feature_dict = dict(zip(unique_feature, list(range(len(unique_feature)))))
        if type(X_train) in [list, np.array, np.ndarray]:
            X_train[:, i] = np.vectorize(unique_feature_dict.get)(X_train[:, i])
            X_val[:, i] = np.vectorize(unique_feature_dict.get)(X_val[:, i])
            X_test[:, i] = np.vectorize(unique_feature_dict.get)(X_test[:, i])
        else:
            column = X_train.columns[i]
            X_train[column] = np.vectorize(unique_feature_dict.get)(X_train[column])
            X_val[column] = np.vectorize(unique_feature_dict.get)(X_val[column])
            X_test[column] = np.vectorize(unique_feature_dict.get)(X_test[column])

    parameter_space = {
        'learning_rate': scipy.stats.loguniform(np.exp(-5), 1.),
        'random_strength': scipy.stats.randint(1, 21),
        'l2_leaf_reg': scipy.stats.loguniform(1., 10.),
        'subsample': scipy.stats.uniform(0., 1.),
        'leaf_estimation_iterations': scipy.stats.randint(1, 11),
        'n_estimators': [n_estimators],
        'cat_features':[cat_features],
        'eval_set': [(X_val, y_val)],
        'logging_level': ['Silent']
    }
    if task_type == 'CPU':
        parameter_space['bootstrap_type'] = 'MVS'
    else:
        parameter_space['bootstrap_type'] = 'Poisson'
    model = CatBoostClassifier(**const_params)
    search = HyperbandSearchCV(model, parameter_space, scoring=roc_auc_scorer)
    search.fit(X_train, y_train)
    print("CatBoost + HyperBand")
    print_results(X_train,
                  X_val,
                  X_test,
                  y_train,
                  y_val,
                  y_test,
                  search)


def choose_smac(X_train,
                X_test,
                X_val,
                y_train,
                y_test,
                y_val,
                cat_features,
                n_estimators=1000,
                max_evals=25,
                task_type="CPU"):
    from ConfigSpace.conditions import InCondition
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
        UniformFloatHyperparameter, UniformIntegerHyperparameter
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.scenario.scenario import Scenario
    cs = ConfigurationSpace()

    learning_rate = UniformFloatHyperparameter("learning_rate", np.exp(-5), 1., log=True)
    l2_leaf_reg = UniformFloatHyperparameter("l2_leaf_reg", 1., 10., log=True)
    subsample = UniformFloatHyperparameter("subsample", 0., 1., log=False)
    random_strength = UniformIntegerHyperparameter("random_strength", 1, 21)
    leaf_estimation_iterations = UniformIntegerHyperparameter("leaf_estimation_iterations", 1, 11)

    cs.add_hyperparameters([learning_rate,
                            random_strength,
                            l2_leaf_reg,
                            subsample,
                            leaf_estimation_iterations])

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": max_evals,
                         # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true"
                         })
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=catboost_from_cfg(X_train,
                                                 X_test,
                                                 X_val,
                                                 y_train,
                                                 y_test,
                                                 y_val,
                                                 cat_features,
                                                 n_estimators,
                                                 max_evals,
                                                 task_type))
    incumbent = smac.optimize()
    inc_value = catboost_from_cfg(X_train,
                                  X_test,
                                  X_val,
                                  y_train,
                                  y_test,
                                  y_val,
                                  cat_features,
                                  n_estimators,
                                  max_evals,
                                  task_type)(incumbent)
    print(inc_value)
    print(dir(incumbent))



def hp_tuning(train, test, validate, column_description, algo_name):
    train_pool = Pool(train, column_description=column_description)
    val_pool = Pool(validate, column_description=column_description)
    test_pool = Pool(test, column_description=column_description)
    #X_train, y_train = train_pool.get_features(), train_pool.get_label()
    #X_val, y_val = val_pool.get_features(), val_pool.get_label()
    #X_test, y_test = test_pool.get_features(), test_pool.get_label()
    cat_features = train_pool.get_cat_feature_indices()

    n_estimators = 1000
    max_evals = 50
    task_type = 'CPU'
    
    #choose_hyperband,
    #choose_smac
    if algo_name == 'classic':
        func = choose_classic
    elif algo_name == 'hyperOpt':
        func = choose_hyperOpt
    elif algo_name == 'gaussian':
        func = choose_gaussian
    elif algo_name == 'optuna':
        func = choose_optuna
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
    results["algo_name"] = algo_name
    results["time"] = end - start
    with open('results.txt', 'w') as outfile:
        json.dump(results, outfile)


import sys
import json
import time

if __name__ == '__main__':
    hp_tuning(sys.argv)

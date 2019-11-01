"""Define XGBoost Class."""
import importlib
from xgboost.sklearn import XGBRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.mongoexp import MongoTrials

from .base_model import BaseModel

#reg_metrics = importlib.import_module("aihc-stats.stats.regression_metrics")


def generate_objective(dataset, model, pbar):
    """
    Train model and return r2 score given current hyperopt params
    """
    (train_X, train_y), (dev_X, dev_y) = dataset.X, dataset.y

    def objective(params):
        pbar.update()
        print(f'Trying {params}')
        m = model()
        m.set_params(**params)
        m.fit(train_X, train_y)
        score = m.score(dev_X, dev_y)
        return 1 - score

    return objective


class XGBoost(BaseModel):
    """XGBoost Class."""

    def __init__(self, XGBoost_objective, tuning_metric, trials='trials',
                 bottom_coding=None, transform=None, **kwargs):
        """Initialize hyperparameters."""
        super(XGBoost, self).__init__(bottom_coding=bottom_coding,
                                      transform=transform)

        self.model = XGBRegressor
        self.tuning_metric = tuning_metric
        self.objective = XGBoost_objective
        self.trials = Trials() \
            if trials == 'trials' \
            else MongoTrials('mongo://localhost:1234/foo_db/jobs',
                             exp_key='exp1')
        self.set_parameters()

    def set_parameters(self):
        self.space = {
            'n_estimators': hp.choice('n_estimators',
                                      list(range(100, 5000, 900))),
            'max_depth': hp.choice('max_depth', list(range(3, 10, 3))),
            'min_child_weight': hp.choice('min_child_weight',
                                          list(range(1, 10, 4))),
            'subsample': hp.choice('subsample', [i / 100.0
                                                 for i in range(75, 100, 10)]),
            'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5, 2)]),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.75, 1, 0.05),
            'objective': self.objective,
            'booster': 'dart',
            'tree_method': 'gpu_exact',
            'n_gpu': 1,
            'silent': 1,
            'learning_rate': 0.1,
            'scale_pos_weight': 1
        }

    def tune(self, training_set, logger=None, saver=None):
        self.training_set = training_set
        objective = generate_objective(self.training_set, self.model)
        best = space_eval(self.space, fmin(fn=objective,
                                           space=self.space,
                                           trials=self.trials,
                                           algo=tpe.suggest,
                                           max_evals=self.max_evals))
        print(f'Best hyperparams: {best}')

        self.model = XGBRegressor()
        self.model.set_params(**best)
        self.model.fit(training_set.X, training_set.y)

    def instantiate_model(self, params):
        model = XGBRegressor()
        model.set_params(**params)
        return model

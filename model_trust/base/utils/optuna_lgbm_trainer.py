import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
import numpy as np

OPTUNA_OPTIMIZE_PARAMS = {"timeout": 11200, "n_trials": 100}
OPTUNA_STUDY_PARAMS = {"direction": "minimize"}

STATIC_MODEL_PARAMS = {}
STATIC_MODEL_PARAMS["classification"] = {
    "objective": "binary",
    "metric": "binary_error",
    "random_state": 42,
}
STATIC_MODEL_PARAMS["regression"] = {
    "objective": "regression",
    "metric": "l2",
    "random_state": 42,
}
STATIC_MODEL_PARAMS["quantile"] = {
    "objective": "quantile",
    "metric": "quantile",
    "alpha": 0.95,
    "random_state": 42,
}


def get_lgbmcv_params(
    trial=None, static_model_params=STATIC_MODEL_PARAMS["regression"]
):
    bounding_params_list = ["min_child_samples", "max_depth"]
    min_child_samples = 50
    max_depth = 5
    if "min_child_samples" in static_model_params.keys():
        min_child_samples = static_model_params["min_child_samples"]
    if "max_depth" in static_model_params.keys():
        max_depth = static_model_params["max_depth"]

    params = {}
    if trial is not None:
        dynamic_params = {
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart"]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", int(np.minimum(1, max_depth)), max_depth
            ),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1e-1),
            "num_boost_round": trial.suggest_int("num_boost_round", 1, 100),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                min_child_samples,
                int(np.maximum(200, min_child_samples)),
            ),
            "random_state": trial.suggest_int("random_state", 40, 50),
        }
        for key in dynamic_params.keys():
            params[key] = dynamic_params[key]

    for key in static_model_params.keys():
        if key not in bounding_params_list:
            params[key] = static_model_params[key]
    # print(" PARAMS :: ")
    # print(params)
    return params


def hp_lgbm_optimizer(
    X,
    Y,
    static_model_params=STATIC_MODEL_PARAMS["regression"],
    optuna_study_params=OPTUNA_STUDY_PARAMS,
    optuna_optimize_params=OPTUNA_OPTIMIZE_PARAMS,
):
    # print("static_model_params :: ")
    # print(static_model_params)

    # print("optuna_study_params :: ")
    # print(optuna_study_params)

    # print("optuna_optimize_params :: ")
    # print(optuna_optimize_params)

    # print()

    optuna_so = optuna_supervised_objective(
        X,
        Y,
        param=lambda x: get_lgbmcv_params(x, static_model_params=static_model_params),
        learner="lgbmcv",
        kfold=5,
        seed=42,
    )
    study = optuna_so.optuna_train(
        optuna_study_params=optuna_study_params,
        optuna_optimize_params=optuna_optimize_params,
    )

    best_params = optuna_so.best_params
    return best_params


class optuna_supervised_objective:
    def __init__(
        self,
        X,
        y,
        param=lambda x: get_lgbmcv_params(x),
        learner="lgbmcv",
        kfold=5,
        seed=42,
    ):
        self.learner = learner
        self.param_fn = param
        self.data = {}
        self.data["X"] = X
        self.data["y"] = y
        self.static_params = self.param_fn(None)

        self.best_params = None
        self.best_value = None
        self.seed = seed
        self.Kfold = KFold(kfold, shuffle=True, random_state=self.seed)

    def objective(self, trial):
        if self.learner in ["lgbmcv"]:
            param = self.param_fn(trial)
            dtrain = lgb.Dataset(self.data["X"], label=self.data["y"])
            # gbm = lgb.cv(param, dtrain, folds=self.Kfold, verbose_eval=False)
            gbm = lgb.cv(param, dtrain, folds=self.Kfold)
            if "metric" in param.keys():
                metric_tag = param["metric"]
            else:
                metric_tag = param["objective"]

            # print("---------")
            for tag in gbm.keys():
                if "mean" in tag:
                    metric_tag = tag.split("-")[0]

            # print(gbm[metric_tag + "-mean"][-1] + 2 * gbm[metric_tag + "-stdv"][-1])

            return gbm[metric_tag + "-mean"][-1] + 2 * gbm[metric_tag + "-stdv"][-1]

    def optuna_train(self, optuna_study_params, optuna_optimize_params=None):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(**optuna_study_params)
        if optuna_optimize_params is not None:
            # print(optuna_optimize_params)
            study.optimize(self.objective, **optuna_optimize_params)
        else:
            study.optimize(self.objective)
        self.best_params = {}
        for key in self.static_params.keys():
            self.best_params[key] = self.static_params[key]
        self.best_params.update(study.best_trial.params)
        #         self.best_params = study.best_trial.params
        self.best_value = study.best_trial.value
        return study


if __name__ == "__main__":
    """
    default main to test correct functionality
    ----------
    """
    import numpy as np

    ## add converter to onnx
    print("Testing :: ")
    X = np.linspace(0, 1, 1000).astype("float")
    Y = np.random.randn(1000) * 0.5 + X * 2
    Y = Y.astype("float")
    X = X[:, np.newaxis]

    print(hp_lgbm_optimizer(X, Y, optuna_optimize_params={"n_trials": 3}))

import itertools
import numpy as np
import optuna

# import sys

from model_trust.base.utils.mt_performance import (
    conformal_percentile,
    get_group_quantile_performance,
)

#### Developing for tree ####
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from model_trust.base.utils.optuna_lgbm_trainer import hp_lgbm_optimizer


def train_quantile_regtree(
    X_train, Y_train, dtree, quantile, X_val=None, Y_val=None, Y_train_surrogate=None
):
    if Y_train_surrogate is None:
        dtree = dtree.fit(X_train, Y_train)
    else:
        dtree = dtree.fit(X_train, Y_train_surrogate)

    if X_val is not None:
        ### Get Leaf conformal quantile estimator ###
        group_prediction = dtree.apply(X_train)
        quantile_group = {}
        for g in np.unique(group_prediction):
            quantile_group[g] = conformal_percentile(
                Y_train[group_prediction == g], quantile * 100
            )

        ### Evaluation ###
        group_prediction_val = dtree.apply(X_val)
        y_pred_val = np.array([quantile_group[g] for g in group_prediction_val])

        quantile_marginal = conformal_percentile(Y_train, quantile * 100)
        performance_dic = get_group_quantile_performance(
            Y_val, y_pred_val, group_prediction_val, quantile_marginal, quantile
        )
        # print()
        # print("performance dic : ", performance_dic)
        return dtree, performance_dic
    else:
        return dtree


def cv_train_quantile_tree(
    X,
    Y,
    dtree,
    quantile,
    surrogate_model=None,
    kfold=KFold(n_splits=5),
    loss_target="coverage_ratio",
    stat="min",
    cv_reduce="2std",
    coverage_tolerance=0.001,
):
    assert loss_target in [
        "pinball_loss",
        "pinball_loss_ratio",
        "coverage",
        "coverage_ratio",
    ], "Target Metric not supported"
    performance_all = {}
    loss_target_list = [loss_target, "coverage"]
    stat_list = [stat, "average"]
    for loss in loss_target_list:
        performance_all[loss] = {}
        for s in stat_list:
            performance_all[loss][s] = []

    groups_stats = {}
    groups_stats["n_groups"] = []
    groups_stats["min_size_group"] = []
    # performance_dic["group"] = group_list
    # performance_dic["count_group"] = count_group
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train = X[train_index]
        Y_train = Y[train_index]

        X_test = X[test_index]
        Y_test = Y[test_index]

        Y_train_surrogate = None
        if surrogate_model is not None:
            # smodel = clone(surrogate_model).fit(X_train, Y_train)
            # Y_train_surrogate = smodel.predict(X_train)
            Y_train_surrogate = surrogate_model.predict(X_train)

        _, performance = train_quantile_regtree(
            X_train,
            Y_train,
            clone(dtree),
            quantile,
            Y_train_surrogate=Y_train_surrogate,
            X_val=X_test,
            Y_val=Y_test,
        )
        for loss in loss_target_list:
            for s in stat_list:
                performance_all[loss][s].append(performance[loss][s])
        groups_stats["n_groups"].append(len(performance["group"]))
        groups_stats["min_size_group"].append(
            np.min(np.array(performance["count_group"]))
        )

    additional_stats = {}
    additional_stats["n_groups"] = np.mean(np.array(groups_stats["n_groups"]))
    additional_stats["min_size_group"] = np.mean(
        np.array(groups_stats["min_size_group"])
    )
    ### Losses (smaller the better)
    if loss_target in ["pinball_loss", "pinball_loss_ratio"]:
        cv_performance = np.array(performance_all[loss_target][stat])

        # print("CV PERFORMANCE :: ", cv_performance)
        # print(
        #     np.mean(cv_performance) + 2 * np.std(cv_performance),
        #     np.mean(cv_performance) + 1 * np.std(cv_performance),
        # )
        # print()

        if np.mean(np.array(performance_all["coverage"]["average"])) >= (
            quantile - coverage_tolerance
        ):
            if cv_reduce == "2std":
                return (
                    np.minimum(
                        np.mean(cv_performance) + 2 * np.std(cv_performance),
                        np.max(cv_performance),
                    ),
                    additional_stats,
                )
            if cv_reduce == "1std":
                return (
                    np.minimum(
                        np.mean(cv_performance) + 1 * np.std(cv_performance),
                        np.max(cv_performance),
                    ),
                    additional_stats,
                )
            if cv_reduce == "max":
                return np.max(cv_performance), additional_stats
        else:
            return np.infty, additional_stats
    ### Coverage (Greater the better)
    if loss_target in ["coverage", "coverage_ratio"]:
        cv_performance = np.array(performance_all[loss_target][stat])

        # print()
        # print("CV PERFORMANCE :: ", cv_performance)
        # print(
        #     np.mean(cv_performance) - 2 * np.std(cv_performance),
        #     np.mean(cv_performance) - 1 * np.std(cv_performance),
        # )
        # print()

        if np.mean(np.array(performance_all["coverage"]["average"])) >= (
            quantile - coverage_tolerance
        ):
            if cv_reduce == "2std":
                return (
                    np.maximum(
                        np.mean(cv_performance) - 2 * np.std(cv_performance),
                        np.min(cv_performance),
                    ),
                    additional_stats,
                )  ### MAXIMUM IS BECAUSE -2std can be below the minimum observed cv
            if cv_reduce == "1std":
                return (
                    np.maximum(
                        np.mean(cv_performance) - 1 * np.std(cv_performance),
                        np.min(cv_performance),
                    ),
                    additional_stats,
                )
            if cv_reduce == "min":
                return np.min(cv_performance), additional_stats

        else:
            return 0, additional_stats


def makeGrid(pars_dict):
    keys = pars_dict.keys()
    combinations = itertools.product(*pars_dict.values())
    ds = [dict(zip(keys, cc)) for cc in combinations]
    return ds


"""
IMPORTANT (BELOW) !! SORT THINGS FROM MORE REGULARIZATION TO LESS REGULARIZATION SUCH THAT FIRST BEST ITEM MODEL HAS LOWEST CAPACITY

"""
QTREE_GRID_SEARCH = {
    "min_samples_leaf": list(
        np.sort(np.array([20, 50, 100, 200, 300, 500, 700, 1000]))[::-1]
    ),
    # "ccp_alpha": [0, 0.0001, 0.0005, 0.001, 0.01, 0.1],
    "ccp_alpha": [0.1, 0.01, 0.001, 0.0005, 0.0001, 0],
    "max_depth": [1, 2, 3, 4, 5, 6],
    "random_state": [42, 43, 44, 45],
    # ""
}


def quantile_tree_optimizer(
    X,
    Y,
    quantile,
    grid_search=QTREE_GRID_SEARCH,
    surrogate_model=None,
    kfold=KFold(n_splits=5),
    loss_target="coverage_ratio",
    stat="min",
    cv_reduce="2std",
    coverage_tolerance=0.01,
    return_best=True,
    min_group_size=20,
):
    """
    Update min leaf size based on min group size
    """
    min_samples_leaf = (
        int(min_group_size * X.shape[0]) if min_group_size < 1 else int(min_group_size)
    )

    min_group_cp_constrain = np.ceil((1 / (1 - quantile)) - 1)
    # print("Min Group Constrain :: ", min_group_cp_constrain)
    min_samples_leaf = np.maximum(min_group_cp_constrain, min_samples_leaf)
    assert (
        min_group_cp_constrain < X.shape[0]
    ), "Number of samples are not sufficient to compute the conformal quantile"
    assert (
        min_samples_leaf < X.shape[0]
    ), "Minimum group size is larger than number of samples"

    # min_samples_leaf_list = []
    # min_samples_leaf_list.append(min_samples_leaf)
    # if "min_samples_leaf" in grid_search.keys():
    #     min_samples_leaf_list_param = grid_search["min_samples_leaf"]
    #     for size in min_samples_leaf_list_param:
    #         if size > min_samples_leaf:
    #             min_samples_leaf_list.append(size)
    # print("MIN SAMPLES LEAF LIST :: ", min_samples_leaf_list)
    # grid_search["min_samples_leaf"] = min_samples_leaf_list

    params_grid = makeGrid(grid_search)
    performance_list = []
    ngroups_list = []
    min_groupsize_list = []
    for params_tree in params_grid:
        dtree = DecisionTreeRegressor(**params_tree)
        performance, additional_stats = cv_train_quantile_tree(
            X,
            Y,
            dtree,
            quantile,
            surrogate_model=surrogate_model,
            kfold=kfold,
            loss_target=loss_target,
            stat=stat,
            cv_reduce=cv_reduce,
            coverage_tolerance=coverage_tolerance,
        )
        # print(
        #     "performance/n_groups/min_size_group : ",
        #     performance,
        #     additional_stats["n_groups"],
        #     additional_stats["min_size_group"],
        # )
        """TODO:: ADD GROUP CONSTRAIN HERE """
        performance_list.append(performance)
        ngroups_list.append(additional_stats["n_groups"])
        min_groupsize_list.append(additional_stats["min_size_group"])

    # print()

    if loss_target in ["pinball_loss", "pinball_loss_ratio"]:
        best_performance_value = np.min(
            np.array(performance_list)[np.array(min_groupsize_list) > min_samples_leaf]
        )
    if loss_target in ["coverage", "coverage_ratio"]:
        best_performance_value = np.max(
            np.array(performance_list)[np.array(min_groupsize_list) > min_samples_leaf]
        )

    best_params = np.array(params_grid)[
        np.array(performance_list) == best_performance_value
    ]  ## Params with best performance
    best_groupsize = np.array(
        np.array(min_groupsize_list)[
            np.array(performance_list) == best_performance_value
        ]
    )  ## best_groupsize

    best_params = best_params[
        np.argmax(best_groupsize)
    ]  ## Choose the one with largest min group size

    performance_dic = {}
    performance_dic["performance_list"] = performance_list
    performance_dic["ngroups_list"] = ngroups_list
    performance_dic["min_groupsize_list"] = min_groupsize_list

    if return_best:
        return best_params
    else:
        return performance_dic, params_grid, best_params


# def get_best_params()


def quantile_lgbm_optimizer(
    X,
    Y,
    quantile,
    optimizer="optuna",
    max_depth=None,
    min_child_samples=None,
    random_state=None,
    timeout=11200,
    n_trials=100,
):
    if optimizer == "optuna":
        QUANTILE_MODEL_STATIC_PARAMS = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": quantile,
        }
        OPTUNA_OPTIMIZE_PARAMS = {"timeout": timeout, "n_trials": n_trials}
        OPTUNA_STUDY_PARAMS = {"direction": "minimize"}

        if random_state is not None:
            QUANTILE_MODEL_STATIC_PARAMS["random_state"] = random_state
        if max_depth is not None:
            QUANTILE_MODEL_STATIC_PARAMS["max_depth"] = max_depth
        if min_child_samples is not None:
            QUANTILE_MODEL_STATIC_PARAMS["min_child_samples"] = min_child_samples
        QUANTILE_MODEL_STATIC_PARAMS["verbose"] = -1
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        # import logging
        # optuna.logging.disable_default_handler()  # Disable the default handler.
        # logger = logging.getLogger()

        # logger.setLevel(logging.ERROR)
        best_params = hp_lgbm_optimizer(
            X,
            Y,
            static_model_params=QUANTILE_MODEL_STATIC_PARAMS,
            optuna_study_params=OPTUNA_STUDY_PARAMS,
            optuna_optimize_params=OPTUNA_OPTIMIZE_PARAMS,
        )

    return best_params

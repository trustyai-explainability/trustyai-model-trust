import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
import logging
import lightgbm as lgb

from model_trust.base.utils.tree_optimization import (
    quantile_lgbm_optimizer,
    quantile_tree_optimizer,
)
from sklearn.tree import DecisionTreeRegressor

LOGGER = logging.getLogger(__name__)


class RegionIdentification:

    """
    Base class for Region (clustering) Discovering/Learning of input features/covariates based on a score variable (e.g., conformal score)
    """

    def __init__(self, params):
        """
        Parameters:
            params (dictionary or list of dictionaries): contains the region identification model parameters to be considered,
                                                         if a list with dictionaries is provided best model search is performed.
        """

        self.params = params
        self.best_params = None  # best model parameters in case that a list of parameter dictionaries were provided
        self.region_model = None  # model
        self.min_group_size = (
            0.1  # if smaller than 1 means fraction if > 1 means number of samples
        )

    def fit(self, X, s, verbose=False):
        """
        Finds and save the best region model from the params list.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
        """
        raise NotImplementedError

    def fit_best(self, X, s, verbose=False):
        """
        Learn the optimal region model by solving an optimization problem.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
        """
        raise NotImplementedError

    # def evaluate(self, X, s):
    # raise NotImplementedError

    def get_regions(self, X):
        """
        Provides the label of the region assigned to each sample in X.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
        """
        raise NotImplementedError

    def evaluate(self, X, s):
        """
        Provides the score performance of the region model.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
        """
        raise NotImplementedError

    def cv_evaluation(self, X, s, n_splits=5, random_state=None, shuffle=False):
        """
        Performs K-fold cross validation and provides the performance scores for each split.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
            n_splits: int corresponding to number of splits
            random_state: int corresponding to random seed for kfold split
            shuffle: bool Shuffle option for kfold spliting
        """
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        cv_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = s[train_index], s[test_index]
            self.fit(X_train, y_train)
            cv_scores.append(self.evaluate(X_test, y_test))
        return cv_scores


class SingleRegion(RegionIdentification):
    """
    Single Region Class, needed for compatibility with standard conformal prediction interval approaches (no groups/regions are considered)
    """

    def __init__(self):
        BaseEstimator.__init__(self)
        self.region_label = 1

    def fit(self, X, s):
        """
        A single region label is considered
        """
        self.region_label = 1
        return self

    def fit_best(self, X, s):
        """
        A single region label is considered
        """
        self.region_label = 1
        return self

    def get_regions(self, X):
        """
        Outputs the single region label that is considered
        """
        return np.ones([X.shape[0]])

    def predict(self, X):
        return np.ones([X.shape[0]])


class RegionQuantileTreeIdentification(RegionIdentification):
    """RegionQuantileTreeIdentification or Multi Region
    
    

    Parameters
    ----------
    params : dict, optional
        Parameters of surrogate model, by default {"random_state": 42}
    quantile : float, optional
        Quantile, by default 0.9
    surrogate_model : str, optional
        Surrogate model, currently only supported model is LGBM, by default "lgbm"
    model_selection_metric : coverage_ratio&quot;, &quot;pinball_loss_ratio&quot;, ], optional
        Model selection metric, can be selected among coverage_ratio, pinball_loss_ratio, by default "coverage_ratio"
    model_selection_stat : min&quot;, &quot;max&quot;, &quot;average&quot;], optional
        Model selection statistics, between min, max, average, by default "min"
    cv_stat : min&quot;, &quot;max&quot;, &quot;1std&quot;, &quot;2std&quot;], optional
        CV Stat, by default "1std"
    min_group_size : int, optional
        Minimum Group Size, by default 20
    """
    def __init__(
        self,
        params={"random_state": 42},
        quantile=0.9,
        surrogate_model="lgbm",
        model_selection_metric: [
            "coverage_ratio",
            "pinball_loss_ratio",
        ] = "coverage_ratio",
        model_selection_stat: ["min", "max", "average"] = "min",
        cv_stat: ["min", "max", "1std", "2std"] = "1std",
        min_group_size=20,
    ):

        self.params = params
        self.quantile = quantile
        # self.model_optimizer = model_optimizer

        if isinstance(params, dict):
            self.best_params = params
            self.init_region_model(self.best_params)
        else:
            self.region_model = None

        self.leaf_values = None

        self.surrogate_model = surrogate_model
        self.surrogate_model_params = None
        self.surrogate_model_instance = None

        self.model_selection_metric = model_selection_metric
        self.model_selection_stat = model_selection_stat
        self.cv_stat = cv_stat
        self.min_group_size = min_group_size

        ### OPTIONS ###
        # 1- no surrogate then RegressionTree Directly, model choice based on metric + group_stat + cv_stat
        # 2- surrogate then quantile gboost then RegressionTree

    def init_region_model(self, params={}):
        """
        Initializes the region identification model, in this case a DecisionTreeRegressor
        ----------
        Parameters:
            params (dictionary): contains a subset of the initialization parameters of region class.
        """

        self.region_model = DecisionTreeRegressor(**params)

    def fit(self, X, s):
        """
        Finds and save the best region model from the params list.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
        """
        return self.fit_best(X, s)

    def fit_surrogate_model_params(self, X, s):
        if self.surrogate_model == "lgbm":
            self.surrogate_model_params = quantile_lgbm_optimizer(
                X, s, quantile=self.quantile
            )
        else:
            raise NotImplementedError


        return self.surrogate_model_params

    def fit_best(self, X, s):
        """
        Performs k-fold cross validation to identify the best set of parameters from self.params,
        then fits the best model with the entire dataset.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            s: np.array containing score variable (e.g., target or conformal score), size (nsamples,)
        """

        smodel = None
        if self.surrogate_model == "lgbm":
            self.surrogate_model_params = self.fit_surrogate_model_params(X, s)
            smodel = lgb.LGBMRegressor(**self.surrogate_model_params)
            smodel = smodel.fit(X, s)
            s_surrogate = smodel.predict(X)
        else:
            s_surrogate = np.array(s)

        self.best_params = quantile_tree_optimizer(
            X,
            s,
            self.quantile,
            surrogate_model=smodel,
            loss_target=self.model_selection_metric,
            stat=self.model_selection_stat,
            cv_reduce=self.cv_stat,
            return_best=True,
            min_group_size=self.min_group_size,
        )

        print("Best Params :" + str(self.best_params))

        self.region_model = DecisionTreeRegressor(**self.best_params)
        print("Best model :" + str(self.region_model))
        self.region_model = self.region_model.fit(X, s_surrogate)

        self.build_membership_translation(X)
        # self.fit(X, s)

        return self

    def build_membership_translation(self, X):
        """
        Generates an array with possible values
        ----------
        """
        self.leaf_values = np.unique(self.region_model.predict(X))

    def get_group_index(self, prediction):
        """
        Returns the updated group label
        ----------
        Parameters:
            prediction: 1D np.array containing the prediction value obtained when using from self.region_model.predict()
        """
        if prediction.shape[0] > 1:
            return np.argmin(
                np.abs(prediction[:, np.newaxis] - self.leaf_values[np.newaxis, :]),
                axis=1,
            )
        else:
            return np.array([np.argmin(np.abs(prediction - self.leaf_values))])

    def get_regions(self, X):
        """
        Provides the label of the region assigned to each sample in X.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
        """
        prediction = self.region_model.predict(X).flatten()
        return np.array(self.get_group_index(prediction))

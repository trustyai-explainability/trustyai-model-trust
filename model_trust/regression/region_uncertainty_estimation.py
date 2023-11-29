import copy
import base64
import logging
import joblib
import numpy as np
from skl2onnx import to_onnx
from model_trust.base.utils.mt_performance import conformal_percentile
from model_trust.base.posthoc_base_uncertainty_estimator import (
    PosthocBaseUncertaintyEstimator,
)
from model_trust.base.region_identification.region_identification_models import (
    SingleRegion,
    RegionQuantileTreeIdentification,
)
from model_trust.base.utils.data_utils import nparray_to_list

LOGGER = logging.getLogger(__name__)


class RegionUncertaintyEstimator(PosthocBaseUncertaintyEstimator):
    """
    This class provides conformal prediction intervals based on uncertainty region discovery for a given point estimate regression model.
    The conformity score we consider is the absolute error |Y-f(X)|, with f the point estimator regression model.

    """

    def __init__(
        self,
        regions_model="single_region",
        regions_model_input_type="covariates",
        confidence: int = 95,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters:
            regions_model: RegionIdentification that defines the type of model that is used to cluster the input
                space into the different uncertainty regions, from the region_identification submodule. Allowed
                values are "single_region", "multi_region". If single_region,
                model_trust.base.region_identification.region_identification_models.SingleRegion is used. If multi_region,
                model_trust.base.region_identification.region_identification_models.RegionQuantileTreeIdentification
                is used. Users can pass parameters to RegionIdentification as keyword arguments such as "multi_region_quantile".
            regions_model_input_type (string): indicate which format to input to the region_model as X. Options are between ["covariates",\
                "predictions"]. Defaults to "covariates"
            random_state (int): random state
            confidence: default confidence interval value.


        """
        self.random_state = random_state

        ### Region identification params ###
        self.regions_model_input_type = regions_model_input_type

        self.confidence = confidence

        self.regions_stats = []
        self.regions_id = {}
        self.regions_model = None
        self.learned_config = {}

        # extract region parameters from kwargs
        region_params = {}
        for arg in kwargs:
            prefix = regions_model + "_"
            if arg.startswith(prefix):
                region_params[arg.removeprefix(prefix)] = kwargs[arg]

        if regions_model == "single_region":
            self.regions_model = SingleRegion(**region_params)
        elif regions_model == "multi_region":
            self.regions_model = RegionQuantileTreeIdentification(**region_params)
        else:
            raise Exception("Region model {} is not supported.".format(regions_model))
        self.regions_model_type = regions_model

    def fit(self, X, y, y_pred):
        """
        Fit a model with the training data and provide prediction for the test set.
        ----------
        Parameters:
            model: ML model class with fit and predict methods
            X: np.array containing input features, size (nsamples, feature dims)
            y: np.array containing target variable, size (nsamples,)
            y_pred: np.array containing predictions from base model, size (nsamples,)

        """

        ### RESET
        self.regions_stats = []
        self.regions_id = {}
        y = y.flatten()  # FORCE 1D DIMENSION O.W. WEIRD CAST HAPPENS

        conformity_scores = np.abs(y - y_pred)  # prefit case

        ## Input for conformal prediction
        X_input, y_pred = self.get_input(X, y_pred)

        ### Fit region identification model
        self.regions_model = self.regions_model.fit_best(X_input, conformity_scores)

        ## Get stats per identified REGION
        membership = self.regions_model.get_regions(X_input)
        ix = 0
        for m in np.sort(np.unique(membership)):
            self.regions_stats.append(conformity_scores[membership == m])
            self.regions_id[str(m)] = ix
            ix += 1

            if np.min(conformity_scores) < 0:
                LOGGER.warning("WARNING CONFORMITY SCORE < 0, for m=" + str(m))
                LOGGER.warning(
                    "min/max/mean conformity "
                    + str(m)
                    + str(np.min(conformity_scores[membership == m]))
                    + str(np.max(conformity_scores[membership == m]))
                    + str(np.mean(conformity_scores[membership == m]))
                )

        self.learned_config["regions_stats"] = self.regions_stats
        self.learned_config["regions_id"] = self.regions_id
        self.learned_config["region_model_type"] = self.regions_model_type
        self.learned_config["confidence"] = self.confidence
        self.learned_config["regions_model_input_type"] = self.regions_model_input_type

        if self.regions_model_type == "multi_region":
            self.learned_config["leaf_values"] = self.regions_model.leaf_values
            onx = to_onnx(
                self.regions_model.region_model, X_input[:1].astype(np.float32)
            )
            onx_ser = onx.SerializeToString()
            self.learned_config["region_model"] = base64.b64encode(onx_ser).decode(
                "ascii"
            )

        return self

    def export_learned_config(self):
        return nparray_to_list(copy.deepcopy(self.learned_config))

    def predict(
        self,
        X,
        y_pred,
        percentile=None,
    ):
        """
        Predict confidence interval
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            y_pred: np.array containing predictions from base model, size (nsamples,)
            percentile: confidence of the prediction interval
        """
        if percentile is None:
            percentile = self.confidence
        X_input, y_pred = self.get_input(X, y_pred)

        ## Get memberships
        membership = self.regions_model.get_regions(X_input)
        interval = []

        # print("PREDICT INTERVAL")
        error_pval = self.get_regions_error_percentile(percentile=percentile)

        for i in range(membership.shape[0]):
            region_i = self.regions_id[str(membership[i])]
            q = error_pval[region_i]
            if q < 0:
                LOGGER.warning("Warning quantile < 0 for sample " + str(i) + str(q))
            interval.append([y_pred[i, 0] - np.abs(q), y_pred[i, 0] + np.abs(q)])
        # print("RETURN")
        return np.array(interval)

    def get_regions(self, X):
        """
        Provides the label of the region assigned to each sample in X.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
        """
        X_input, _ = self.get_input(X)
        return self.regions_model.get_regions(X_input)

    def get_regions_error_percentile(self, percentile=None):
        """
        Provides the conformal percentiles for each uncertainty region
        ----------
        Parameters:
            percentile: np.array containing percentile to compute
        """
        if percentile is None:
            percentile = self.confidence
        error_pval = np.zeros([len(self.regions_stats)])
        for m in range(len(self.regions_stats)):
            error_pval[m] = conformal_percentile(self.regions_stats[m], percentile)
            if error_pval[m] < 0:
                LOGGER.warning(
                    "WARNING PERCENTILE IS < 0"
                    + str(error_pval[m])
                    + "; region "
                    + str(m)
                )
                LOGGER.warning(
                    "min/max/avg :"
                    + str(np.min(self.regions_stats[m]))
                    + str(np.max(self.regions_stats[m]))
                    + str(np.mean((self.regions_stats[m])))
                )
        return error_pval

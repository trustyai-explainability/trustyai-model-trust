import numpy as np
import pandas as pd
import abc
import torch


class PosthocBaseUncertaintyEstimator:
    """
    Base class for posthoc uncertainty estimation of a point estimate machine learning model \
    This is a generic class and allows different option for the uncertainty estimator \
    
    Setting:
        Given a point estimate regression model f: X -> Y (pretrained or not),
        the objective of this base class is to provide a confidence interval/set estimator model
        based on f. (e.g., for a given X provide a confidence interval/set CI for confidence level L
                     CI(X)=[u(X),v(X)] (regression) or CI(X)= \{c_1,...,c_k\} (classification) such that p(Y \in CI(X)) = L)
    """

    def __init__(
        self,
        regions_model_input_type="covariates",
        task="regression",
    ):
        """
        Parameters:
            regions_model_input_type (string): indicate if the input to the regions_model should be "covariates" or "predictions". Defaults to "covariates.
        """

        self.regions_model_input_type = regions_model_input_type
        self.task = task
        if (self.regions_model_input_type != "covariates") and (
            self.regions_model_input_type != "predictions"
        ):
            self.regions_model_input_type = "predictions"

    @abc.abstractmethod
    def init_uncertainty_model(self, *argv, **kwargs):
        """
        Initialize the uncertainty model.
        """
        raise NotImplementedError

    def get_input(self, X, y_pred):
        """
        Generates the inputs for the confidence interval/set estimation model
        --------
        Parameters:
            X : input covariates
            y_pred: np.array containing predictions from base model, size (nsamples,)
        """

        if len(y_pred.shape) == 1:
            y_pred = y_pred[:, np.newaxis]

        X_input = None
        if self.regions_model_input_type == "covariates":
            X_input = np.array(X)

        if self.regions_model_input_type == "predictions":
            if X_input is not None:
                X_input = np.concatenate([X_input, y_pred], axis=1)
            else:
                X_input = y_pred

        return X_input, y_pred

    def fit(self, X, y, y_pred, verbose=False):
        """
        Learn the confidence interval/set estimation model.
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            y: np.array containing target variable, size (nsamples,)
        """
        raise NotImplementedError

    def predict_with_uncertainty(self, X, verbose=True):
        """
        Predicts the uncertainty for a set of datapoints
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
        """
        # Returns y_pred, uncertainty_params

        raise NotImplementedError

    def predict(self, X, y_pred):
        """
        Point estimate prediction of the ML model
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
        """
        raise NotImplementedError

    def predict_interval(self, X, percentile=95, verbose=True):
        """
        Predict confidence interval
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            percentile: confidence of the prediction interval
        """
        raise NotImplementedError

    def get_calibration_table(self, X, Y, verbose=False):
        """
        Generate a calibration table (confidence vs coverage) for the learned confidence interval/set model
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            Y: np.array containing target variable, size (nsamples, )
        """
        row = []
        if self.task == "regression":
            for conf in np.linspace(0, 100, 11)[1:-1]:
                intervals = self.predict_interval(X, percentile=conf, verbose=verbose)
                acc = np.mean((Y <= intervals[:, 1]) & (Y >= intervals[:, 0]))
                row.append([conf, acc * 100])
            return pd.DataFrame(data=row, columns=["confidence", "coverage"])

    ## New Method for evaluation
    def get_ci_performance_table(
        self, X, Y, percentiles=list(np.linspace(0, 100, 11)[1:-1]), verbose=False
    ):
        """
        Generate a performance table for the learned confidence interva/set model
        ----------
        Parameters:
            X: np.array containing input features, size (nsamples, feature dims)
            Y: np.array containing target variable, size (nsamples, )
            percentiles: list containing confidence level to consider
        """

        row = []
        if self.task == "regression":
            for conf in percentiles:
                intervals = self.predict_interval(X, percentile=conf, verbose=verbose)
                coverage = np.mean((Y <= intervals[:, 1]) & (Y >= intervals[:, 0]))
                row.append(
                    [
                        conf / 100,
                        coverage,
                        np.median(np.abs(intervals[:, 1] - intervals[:, 0])),
                        np.mean(np.abs(intervals[:, 1] - intervals[:, 0])),
                        coverage - conf / 100,
                    ]
                )
            return pd.DataFrame(
                data=row,
                columns=[
                    "confidence",
                    "coverage",
                    "ci_median_width",
                    "ci_mean_width",
                    "diff_cov_conf",
                ],
            )

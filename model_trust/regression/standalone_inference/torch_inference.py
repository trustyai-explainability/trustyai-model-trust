import torch
import numpy as np


class CPInferenceSingleRegion(torch.nn.Module):
    ### Base Torch Module to perform conformal prediction interval
    def __init__(self, conformity_scores, quantile=-1):
        """
        Base Torch Module to do forward computation of conformal prediction interval
        ----------
        Parameters:
            Conformity scores: 1D numpy array with conformity scores
            Quantile : real value in [0,1]
        """
        ### Conformity scores: 1D numpy array with conformity scores
        ### Quantile : float in [0,1] or outside the interval to store entire conformity vector.

        # print("HERE")
        super().__init__()
        if isinstance(conformity_scores, np.ndarray):
            conformity_scores = torch.from_numpy(conformity_scores)
            conformity_scores = torch.tensor(conformity_scores, dtype=torch.float)

        self.quantile = torch.tensor(quantile, dtype=torch.float)
        self.quantile_stored = False
        # print("conformity_scores :", conformity_scores)
        # print("quantile :", self.quantile)

        if (self.quantile > 1) or (self.quantile < 0):
            raise Exception("quantile must be a real value in [0,1].")

        self.conformity_scores = self._conformal_quantile(
            conformity_scores, self.quantile
        )

        self.conformity_scores = torch.nn.Parameter(
            self.conformity_scores, requires_grad=False
        )
        self.quantile = torch.nn.Parameter(self.quantile, requires_grad=False)

    def _conformal_quantile(self, scores, quantile):
        """
        conformal quantile estimation
        ----------
        Parameters:
            scores: torch tensor containing the scores, size (nsamples,)
            percentile: quantile to compute
        """
        n = scores.shape[0]
        q = torch.ceil((n + 1) * quantile) / n
        q = torch.minimum(q, torch.tensor(1, dtype=torch.float))
        qhat = torch.quantile(scores, q, interpolation="higher")
        # print("TORCH QUANTILE : ", qhat)
        return qhat

    def forward(self, prediction):
        ## prediction: shape = nsamples x 1
        if len(prediction) == 1:
            prediction = prediction.unsqueeze(1)
        return (
            prediction,
            (prediction - self.conformity_scores),
            (prediction + self.conformity_scores),
        )


class CPInferenceMultiRegion(torch.nn.Module):
    ### Base Torch Module to perform conformal prediction interval
    def __init__(self, conformity_scores_list, leaf_values, quantile=-1):
        """
        Base Torch Module to do forward computation of conformal prediction interval
        ----------
        Parameters:
            Conformity scores: 1D numpy array with conformity scores
            Quantile : float in [0,1] or outside the interval to store entire conformity scores as parameters.
        """
        ### Conformity scores: 1D numpy array with conformity scores
        ### Quantile : float in [0,1] or outside the interval to store entire conformity vector.

        # print("HERE")
        super().__init__()
        conformity_scores_list = [
            torch.tensor(region_stats, dtype=torch.float)
            for region_stats in conformity_scores_list
        ]

        quantile = torch.tensor(quantile, dtype=torch.float)
        leaf_values = torch.tensor(leaf_values, dtype=torch.float)

        if (quantile > 1) or (quantile < 0):
            raise Exception("quantile must be a real value in [0,1].")

        # compute effective conformity scores
        error_pval = [
            self._conformal_quantile(conformity_scores, quantile)
            for conformity_scores in conformity_scores_list
        ]
        error_pval = torch.tensor(error_pval, dtype=torch.float)
        error_pval = error_pval.unsqueeze(1)

        self.error_pval = torch.nn.Parameter(error_pval, requires_grad=False)
        self.leaf_values = torch.nn.Parameter(leaf_values, requires_grad=False)
        self.quantile = torch.nn.Parameter(quantile, requires_grad=False)

    def _conformal_quantile(self, scores, quantile):
        """
        conformal quantile estimation
        ----------
        Parameters:
            scores: torch tensor containing the scores, size (nsamples,)
            percentile: quantile to compute
        """
        n = scores.shape[0]
        q = torch.ceil((n + 1) * quantile) / n
        q = torch.minimum(q, torch.tensor(1, dtype=torch.float))
        qhat = torch.quantile(scores, q, interpolation="higher")
        return qhat

    def forward(self, x_orig, base_prediction, region_prediction):
        ## prediction: shape = nsamples x 1

        membership = torch.argmin(
            torch.abs(region_prediction - self.leaf_values), axis=1
        )

        return (
            x_orig,
            base_prediction,
            region_prediction,
            base_prediction - self.error_pval[membership],
            base_prediction + self.error_pval[membership],
        )

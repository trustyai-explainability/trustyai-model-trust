import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

"""
The inference function cp_inference (this file) must be isolated from rest of the
model_trust code. So, it should not import any components from model_trust.
"""


def cp_inference(learned_config, X, y_pred, percentile=None):
    interval = []
    if percentile is None:
        percentile = learned_config["confidence"]
    if len(y_pred.shape) == 1:
        y_pred = y_pred[:, np.newaxis]

    # input preparation
    X_input = None
    if learned_config["regions_model_input_type"] == "covariates":
        X_input = np.array(X)

    if learned_config["regions_model_input_type"] == "predictions":
        if X_input is not None:
            X_input = np.concatenate([X_input, y_pred], axis=1)
        else:
            X_input = y_pred

    # get regions
    membership = []
    if learned_config["region_model_type"] == "single_region":
        membership = np.ones([X_input.shape[0]])
    elif learned_config["region_model_type"] == "multi_region":
        prediction = learned_config["region_model"].predict(X_input).flatten()
        if prediction.shape[0] > 1:
            membership = np.argmin(
                np.abs(
                    prediction[:, np.newaxis]
                    - learned_config["leaf_values"][np.newaxis, :]
                ),
                axis=1,
            )
        else:
            membership = np.array(
                [np.argmin(np.abs(prediction - learned_config["leaf_values"]))]
            )
    else:
        raise Exception(
            "Region model type {} is not supported in inference.".format(
                learned_config["region_model_type"]
            )
        )  # this is only in case of manipulated config

    # get region percentile error
    error_pval = np.zeros([len(learned_config["regions_stats"])])
    for m in range(len(learned_config["regions_stats"])):
        scores = np.array(learned_config["regions_stats"][m])
        n = scores.shape[0]
        q = np.ceil((n + 1) * (percentile / 100)) / n
        q = np.minimum(q, 1)
        error_pval[m] = np.quantile(scores, q, method="higher")

        if error_pval[m] < 0:
            LOGGER.warning(
                "WARNING PERCENTILE IS < 0" + str(error_pval[m]) + "; region " + str(m)
            )
            LOGGER.warning(
                "min/max/avg :"
                + str(np.min(learned_config["regions_stats"][m]))
                + str(np.max(learned_config["regions_stats"][m]))
                + str(np.mean((learned_config["regions_stats"][m])))
            )
    for i in range(membership.shape[0]):
        region_i = learned_config["regions_id"][membership[i]]
        q = error_pval[region_i]
        if q < 0:
            LOGGER.warning("Warning quantile < 0 for sample " + str(i) + str(q))
        interval.append([y_pred[i, 0] - np.abs(q), y_pred[i, 0] + np.abs(q)])
    return np.array(interval)

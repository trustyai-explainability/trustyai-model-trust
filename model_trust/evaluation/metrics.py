import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_pinball_loss


def get_interval_metrics(y_true, y_mean, y_interval):
    mse = mean_squared_error(y_true, y_mean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_mean)

    coverage = np.mean(
        (y_true[:] >= y_interval[:, 0]) & (y_true[:] <= y_interval[:, 1])
    )
    mean_width = (np.abs(y_interval[:, 1] - y_interval[:, 0])).mean()
    median_width = np.median(np.abs(y_interval[:, 1] - y_interval[:, 0]))

    return {
        "rmse": rmse,
        "r2": r2,
        "coverage": coverage,
        "mean_width": mean_width,
        "median_width": median_width,
    }


def get_group_metrics(y_lower, y_upper, y_pred, y_gt, group_pred=None, confidence=0.95):
    """Get group metrics

    Args:
        y_lower (np.array or pd.Series): Lower bound predictions of y
        y_upper (np.array or pd.Series): Upper bound predictions of y
        y_pred (np.array or pd.Series): Predictions
        y_gt (np.array or pd.Series): Ground truth labels
        group_pred (np.array or pd.Series, optional): Group predictions. Defaults to None.
        confidence (float, optional): Confidence threshold. Defaults to 0.95.

    Returns:
        pd.DataFrame: Metrics dictionary with evaluation metrics
    """
    y_interval = np.concatenate(
        [y_lower[:, np.newaxis], y_upper[:, np.newaxis]], axis=1
    )

    if group_pred is None:
        group_label = np.ones([y_gt.shape[0]])
    else:
        # group_label = group_pred.values '''this gave me an error'''
        group_label = group_pred

    rows = []
    for g in np.unique(group_label):
        rows_g = [g]
        out_tag = get_interval_metrics(
            y_gt[group_label == g],
            y_pred[group_label == g],
            y_interval[group_label == g],
        )
        for tag in out_tag.keys():
            rows_g.append(out_tag[tag])

        pb_loss = mean_pinball_loss(
            np.abs(y_gt[group_label == g] - y_pred[group_label == g]),
            np.abs(y_upper[group_label == g] - y_lower[group_label == g]),
            alpha=confidence,
        )
        rows_g.append(pb_loss)
        rows_g.append(np.sum(group_label == g))
        rows_g.append(np.sum(group_label == g) / group_label.shape[0])

        rows.append(rows_g)

    columns = ["group"]
    columns.extend([_ for _ in out_tag.keys()])
    columns.append("PBloss")
    columns.append("group_samples")
    columns.append("group_size")
    return pd.DataFrame(data=rows, columns=columns)

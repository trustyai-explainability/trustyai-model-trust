from sklearn.metrics import mean_pinball_loss
import numpy as np


def conformal_percentile(scores, percentile):
    """
    conformal quantile estimation
    ----------
    Parameters:
        scores: np.array containing the scores, size (nsamples,)
        percentile: quantile to compute
    """
    scores = np.array(scores)
    n = scores.shape[0]
    q = np.ceil((n + 1) * (percentile / 100)) / n
    q = np.minimum(q, 1)
    qhat = np.quantile(scores, q, method="higher")

    return qhat


def CP_metrics(
    y_true, y_pred, quantile, metric="pinball_loss"
):  # ,quantile_marginal = None):
    # if quantile_marginal is None:
    # quantile_marginal = conformal_percentile(y_pred,quantile*100)
    # quantile_marginal = quantile_marginal*np.ones_like(y_true)
    if metric == "pinball_loss":
        return mean_pinball_loss(y_true, y_pred, alpha=quantile)
    # if metric == 'pinball_loss_ratio':
    #     numerator = mean_pinball_loss(y_true, y_pred ,alpha = quantile)
    #     denominator = mean_pinball_loss(y_true, quantile_marginal ,alpha = quantile)
    #     return numerator/denominator
    if metric == "coverage":
        return np.mean(y_true <= y_pred)
    # if metric == 'coverage_ratio':
    #     numerator = np.mean(y_true<=y_pred)
    #     denominator = np.mean(y_true<=quantile_marginal)
    #     return numerator/np.minimum(denominator,quantile)


def get_group_quantile_performance(
    y_true, y_pred, group, quantile_marginal, quantile, summarized_metrics=True
):
    loss_pinball_group = []
    count_group = []
    coverage_group = []
    # loss_pinball_ratio_group = []
    # coverage_ratio_group = []

    loss_pinball_baseline_group = []
    coverage_baseline_group = []
    group_list = []

    for g in np.unique(group):
        ngroup = np.sum(group == g)
        count_group.append(ngroup)
        coverage_group.append(
            CP_metrics(
                y_true[group == g],
                y_pred[group == g],
                quantile,
                # quantile_marginal=quantile_marginal,
                metric="coverage",
            )
        )

        coverage_baseline_group.append(
            CP_metrics(
                y_true[group == g],
                np.ones_like(y_pred[group == g]) * quantile_marginal,
                quantile,
                # quantile_marginal=quantile_marginal,
                metric="coverage",
            )
        )

        # coverage_ratio_group.append(CP_metrics(y_true[group==g],y_pred[group==g],
        #                                  quantile,
        #                                  quantile_marginal = quantile_marginal,metric = 'coverage_ratio'))

        loss_pinball_group.append(
            CP_metrics(
                y_true[group == g],
                y_pred[group == g],
                quantile,
                # quantile_marginal=quantile_marginal,
                metric="pinball_loss",
            )
        )

        loss_pinball_baseline_group.append(
            CP_metrics(
                y_true[group == g],
                np.ones_like(y_pred[group == g]) * quantile_marginal,
                quantile,
                # quantile_marginal=quantile_marginal,
                metric="pinball_loss",
            )
        )

        # loss_pinball_ratio_group.append(CP_metrics(y_true[group==g],y_pred[group==g],
        #                                  quantile,
        #                                  quantile_marginal = quantile_marginal,metric = 'pinball_loss_ratio'))

        group_list.append(g)

    performance_dic = {}
    performance_dic["group"] = group_list
    performance_dic["count_group"] = count_group
    performance_dic["coverage"] = coverage_group
    # performance_dic["coverage_ratio"] = coverage_ratio_group
    performance_dic["coverage_baseline"] = coverage_baseline_group
    performance_dic["pinball_loss"] = loss_pinball_group
    performance_dic["pinball_loss_baseline"] = loss_pinball_baseline_group
    # performance_dic["pinball_loss_ratio"] = loss_pinball_ratio_group

    if summarized_metrics:
        summarized_dic = {}
        summarized_dic["group"] = group_list
        summarized_dic["count_group"] = count_group
        for metric in [
            "coverage",
            "coverage_baseline",
            "pinball_loss",
            "pinball_loss_baseline",
        ]:
            values = np.array(performance_dic[metric])
            # print(metric, values)
            summarized_dic[metric] = {}
            count_group = np.array(performance_dic["count_group"])
            summarized_dic[metric]["min"] = np.min(values)
            summarized_dic[metric]["max"] = np.max(values)
            summarized_dic[metric]["average"] = np.sum(count_group * values) / np.sum(
                count_group
            )
            summarized_dic[metric]["balance"] = np.mean(values)
        for metric in ["coverage", "pinball_loss"]:
            summarized_dic[metric + "_ratio"] = {}
            summarized_dic[metric + "_ratio"]["average"] = (
                summarized_dic[metric]["average"]
                / summarized_dic[metric + "_baseline"]["average"]
            )
            summarized_dic[metric + "_ratio"]["balance"] = (
                summarized_dic[metric]["balance"]
                / summarized_dic[metric + "_baseline"]["balance"]
            )
            summarized_dic[metric + "_ratio"]["max"] = (
                summarized_dic[metric]["max"]
                / summarized_dic[metric + "_baseline"]["max"]
            )
            summarized_dic[metric + "_ratio"]["min"] = (
                summarized_dic[metric]["min"]
                / summarized_dic[metric + "_baseline"]["min"]
            )

        return summarized_dic

    else:
        return performance_dic

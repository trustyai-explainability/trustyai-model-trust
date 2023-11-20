from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_linear_2_region_data(
    nsamples=3000,
    quantile=0.9,
    sigma_0=1,
    sigma_1=4,
    test_ratio=0.2,
    cal_ratio=0.2,
    seed=42,
):
    np.random.seed(seed)
    group_function = lambda x: x >= 0
    y_mean_function = lambda x: 10 * x
    sigmas_around_mean = lambda p: (norm.ppf(0.5 + p / 2) - norm.ppf(0.5 - p / 2)) / 2

    def get_pintervals(X):
        G = group_function(X)
        Y_mean = y_mean_function(X)

        pi_gt_high = (sigma_0 * sigmas_around_mean(quantile) + Y_mean) * (G == 0) + (
            sigma_1 * sigmas_around_mean(quantile) + Y_mean
        ) * (G == 1)
        pi_gt_low = (-sigma_0 * sigmas_around_mean(quantile) + Y_mean) * (G == 0) + (
            -sigma_1 * sigmas_around_mean(quantile) + Y_mean
        ) * (G == 1)

        return pi_gt_low, pi_gt_high

    """ GENERATE DATA"""
    X = np.random.random(nsamples) - 0.5
    Y_mean = y_mean_function(X)
    G = group_function(X)

    Y = (
        Y_mean
        + sigma_0 * np.random.randn(nsamples) * (G == 0)
        + sigma_1 * np.random.randn(nsamples) * (G == 1)
    )

    pi_gt_low, pi_gt_high = get_pintervals(X)

    # pi_gt_high = (sigma_0*sigmas_around_mean(quantile) + Y_mean)*(G==0) + (sigma_1*sigmas_around_mean(quantile) + Y_mean)*(G == 1)
    # pi_gt_low = (-sigma_0*sigmas_around_mean(quantile) + Y_mean)*(G==0) + (-sigma_1*sigmas_around_mean(quantile) + Y_mean)*(G == 1)

    # coverage_empirical = np.mean((Y <= pi_gt_high) & (Y >= pi_gt_low))

    X_data = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)
    X_data = np.concatenate([X_data, Y_mean[:, np.newaxis]], axis=1)
    X_data = np.concatenate([X_data, G[:, np.newaxis]], axis=1)
    X_data = np.concatenate([X_data, pi_gt_low[:, np.newaxis]], axis=1)
    X_data = np.concatenate([X_data, pi_gt_high[:, np.newaxis]], axis=1)

    pd_data = pd.DataFrame(
        data=X_data, columns=["X", "Y", "Y_mean", "G", "PI_GT_LOW", "PI_GT_HIGH"]
    )

    features = ["X"]
    target = "Y"

    X = pd_data[features].values
    Y = pd_data[target].values
    X_tr, X_test, Y_tr, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=seed, shuffle=True
    )
    X_train, X_cal, Y_train, Y_cal = train_test_split(
        X_tr, Y_tr, test_size=cal_ratio, random_state=seed, shuffle=True
    )

    Y_train = Y_train.squeeze()
    Y_test = Y_test.squeeze()
    Y_cal = Y_cal.squeeze()

    data_dic = {}
    data_dic["data"] = pd_data
    data_dic["features"] = features
    data_dic["target"] = target
    data_dic["x_train"] = X_train
    data_dic["y_train"] = Y_train
    data_dic["x_test"] = X_test
    data_dic["y_test"] = Y_test
    data_dic["x_cal"] = X_cal
    data_dic["y_cal"] = Y_cal

    data_dic["y_mean_function"] = y_mean_function
    data_dic["get_pintervals_function"] = get_pintervals

    return data_dic

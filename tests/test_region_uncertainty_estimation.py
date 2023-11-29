import unittest
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from model_trust.evaluation.metrics import get_group_metrics
from model_trust.datasets.synthetic_data import load_linear_2_region_data
from model_trust.regression.region_uncertainty_estimation import (
    RegionUncertaintyEstimator,
)
from model_trust.regression.standalone_inference import cp_inference


class TestRegionUncertaintyEstimator(unittest.TestCase):
    def setUp(self):
        # load data
        self.quantile = 0.9
        dataset = load_linear_2_region_data(
            nsamples=3000,
            quantile=self.quantile,
            sigma_0=1,
            sigma_1=4,
            test_ratio=0.2,
            cal_ratio=0.2,
            seed=42,
        )

        data = dataset["data"]
        features = dataset["features"]
        target = dataset["target"]
        X = data[features].values
        Y = data[target].values

        x_train = dataset["x_train"]
        self.x_test = dataset["x_test"]
        self.x_cal = dataset["x_cal"]

        y_train = dataset["y_train"]
        self.y_test = dataset["y_test"]
        self.y_cal = dataset["y_cal"]

        pi_gt_low = data["PI_GT_LOW"].values
        pi_gt_high = data["PI_GT_HIGH"].values
        Y_mean = data["Y_mean"].values
        coverage_empirical = np.mean((Y <= pi_gt_high) & (Y >= pi_gt_low))

        X_sort = np.sort(X.flatten())
        X_sort_indices = np.argsort(X.flatten())

        base_model = LinearRegression().fit(x_train, y_train)
        self.y_pred_cal = base_model.predict(self.x_cal)
        self.y_pred = base_model.predict(self.x_test)

    def test_single_region(self):
        # initialize/fit region uncertainty estimator for single region

        cp_config_input = {}
        cp_config_input["confidence"] = self.quantile * 100
        cp_config_input["regions_model"] = "single_region"

        # single region does not have region parameters
        cp_single_region_model = RegionUncertaintyEstimator(**cp_config_input)
        cp_single_region_model.fit(self.x_cal, self.y_cal, self.y_pred_cal)

        # export learned config after fit.
        cp_single_region_model_learned_params = (
            cp_single_region_model.export_learned_config()
        )

        # store learned config in json file
        with open("cp_single_region_model_learned_params.json", "w") as fp:
            json.dump(cp_single_region_model_learned_params, fp)

        # load config from json file
        with open("cp_single_region_model_learned_params.json", "r") as fp:
            cp_single_region_model_learned_params_from_json = json.load(fp)

        pred_intervals = cp_inference(
            learned_config=cp_single_region_model_learned_params_from_json,
            X=self.x_test[0:1],
            y_pred=self.y_pred[0:1],
            percentile=90,
        )
        pred_intervals = [
            [round(num, 8) for num in interval] for interval in pred_intervals
        ]

        # validate computed prediction intervals for 1st test point
        self.assertIsNotNone(pred_intervals)
        self.assertListEqual(pred_intervals, [[-2.44483532, 7.28648244]])

    def test_multi_region(self):
        # initialize/fit region uncertainty estimator for multi region

        cp_init_params = {}
        cp_init_params["confidence"] = self.quantile * 100
        cp_init_params["regions_model"] = "multi_region"

        # multi region parameters
        cp_init_params["multi_region_model_selection_metric"] = "coverage_ratio"
        cp_init_params["multi_region_model_selection_stat"] = "min"
        cp_init_params["multi_region_min_group_size"] = 20

        cp_multi_region_model = RegionUncertaintyEstimator(**cp_init_params)
        cp_multi_region_model.fit(self.x_cal, self.y_cal, self.y_pred_cal)

        # export learned config after fit.
        cp_multi_region_model_learned_params = (
            cp_multi_region_model.export_learned_config()
        )

        # store learned config in json file
        with open("cp_multi_region_model_learned_params.json", "w") as fp:
            json.dump(cp_multi_region_model_learned_params, fp)

        # load config from json file
        with open("cp_multi_region_model_learned_params.json", "r") as fp:
            cp_multi_region_model_learned_params_from_json = json.load(fp)

        # test inference
        pred_intervals = cp_inference(
            learned_config=cp_multi_region_model_learned_params_from_json,
            X=self.x_test[0:1],
            y_pred=self.y_pred[0:1],
        )

        pred_intervals = [
            [round(num, 1) for num in interval] for interval in pred_intervals
        ]

        # validate computed prediction intervals for 1st test point
        self.assertIsNotNone(pred_intervals)
        self.assertListEqual(pred_intervals, [[-4.6, 9.4]])


if __name__ == "__main__":
    unittest.main()

import unittest
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump
import onnxruntime as rt
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from model_trust.datasets.synthetic_data import load_linear_2_region_data
from model_trust.regression.region_uncertainty_estimation import (
    RegionUncertaintyEstimator,
)


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
        OPSET_VERSION = 17
        base_onnx_model = to_onnx(
            base_model,
            x_train[:1].astype(np.float32),
            target_opset=OPSET_VERSION,
            # final_types=[('Y_Pred', FloatTensorType([None, 1]))],
        )
        self.base_onnx_model_str = base_onnx_model.SerializeToString()

    def test_single_region(self):
        # initialize/fit region uncertainty estimator for single region

        single_region_cp_inputs = {}
        single_region_cp_inputs["base_model"] = self.base_onnx_model_str
        single_region_cp_inputs["confidence"] = self.quantile * 100
        single_region_cp_inputs["regions_model"] = "single_region"

        # single region does not have region parameters
        cp_single_region_model = RegionUncertaintyEstimator(**single_region_cp_inputs)
        cp_single_region_model.fit(self.x_cal, self.y_cal, self.y_pred_cal)

        # export learned config after fit.
        cp_single_region_model_learned_params = (
            cp_single_region_model.export_learned_config()
        )

        # dump onnx model to a file
        with open("single_region_cp_model.onnx", "wb") as fp:
            fp.write(cp_single_region_model_learned_params["combined_model"])

        # load the onnx model into memory
        with open("single_region_cp_model.onnx", "rb") as fp:
            single_region_cp_model_bytes = fp.read()

        single_region_sess = rt.InferenceSession(single_region_cp_model_bytes)

        single_region_model_onnx_output = single_region_sess.run(
            None, {"X": self.x_test[0:1].astype(np.float32)}
        )

        # validate computed prediction intervals for 1st test point
        self.assertIsNotNone(single_region_model_onnx_output)
        # self.assertListEqual(pred_intervals, [[-2.44483532, 7.28648244]])

    def test_multi_region(self):
        # initialize/fit region uncertainty estimator for multi region

        cp_init_params = {}
        cp_init_params["confidence"] = self.quantile * 100
        cp_init_params["regions_model"] = "multi_region"
        cp_init_params["base_model"] = self.base_onnx_model_str

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

        # dump onnx model to a file
        with open("multi_region_cp_model.onnx", "wb") as fp:
            fp.write(cp_multi_region_model_learned_params["combined_model"])

        # load the onnx model into memory
        with open("multi_region_cp_model.onnx", "rb") as fp:
            multi_region_cp_model_bytes = fp.read()

        # test inference
        multi_region_sess = rt.InferenceSession(multi_region_cp_model_bytes)

        multi_region_model_onnx_output = multi_region_sess.run(
            None, {"X": self.x_test[0:1].astype(np.float32)}
        )

        # validate computed prediction intervals for 1st test point
        self.assertIsNotNone(multi_region_model_onnx_output)
        # self.assertListEqual(pred_intervals, [[-4.6, 9.4]])


if __name__ == "__main__":
    unittest.main()

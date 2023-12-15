import io
import copy
import base64
import logging
import joblib
import onnx
import torch
import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx
from onnx.compose import merge_models
from model_trust.base.utils.mt_performance import conformal_percentile
from model_trust.base.posthoc_base_uncertainty_estimator import (
    PosthocBaseUncertaintyEstimator,
)
from model_trust.base.region_identification.region_identification_models import (
    SingleRegion,
    RegionQuantileTreeIdentification,
)
from model_trust.base.utils.data_utils import nparray_to_list
from model_trust.regression.standalone_inference.torch_inference import (
    CPInferenceSingleRegion,
    CPInferenceMultiRegion,
)

LOGGER = logging.getLogger(__name__)


class RegionUncertaintyEstimator(PosthocBaseUncertaintyEstimator):
    """
    This class provides conformal prediction intervals based on uncertainty region discovery for a given point estimate regression model.
    The conformity score we consider is the absolute error |Y-f(X)|, with f the point estimator regression model.

    """

    def __init__(
        self,
        base_model: bytes,
        regions_model: str = "single_region",
        regions_model_input_type: str = "covariates",
        confidence: int = 95,
        random_state: int = 42,
        opset_version: int = 17,
        **kwargs,
    ):
        """
        Parameters:
            base_model: ONNX serialized model string converted to bytes using .SerializeToString(). Refer to 
                https://onnx.ai/onnx/api/serialization.html#onnx.ModelProto for example usage.
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
        self.base_model = base_model
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
        self.opset_version = opset_version

    def fit(self, X, y):
        """
        Fit a model with the training data and provide prediction for the test set.
        ----------
        Parameters:
            model: ML model class with fit and predict methods
            X: np.array containing input features, size (nsamples, feature dims)
            y: np.array containing target variable, size (nsamples,)

        """

        ### RESET
        self.regions_stats = []
        self.regions_id = {}
        y = y.flatten()  # FORCE 1D DIMENSION O.W. WEIRD CAST HAPPENS

        # compute base model predictions
        base_model_sess = rt.InferenceSession(self.base_model)
        y_pred = base_model_sess.run(None, {"X": X.astype(np.float32)})[0].flatten()
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

        # torch based onnx serialization
        torch_input_name = "input_x"
        torch_output_names = [
            "prediction",
            "exp_lower_pred_endpoint",
            "exp_upper_pred_endpoint",
        ]

        # old code snippet
        self.learned_config["regions_stats"] = self.regions_stats
        self.learned_config["regions_id"] = self.regions_id
        self.learned_config["region_model_type"] = self.regions_model_type
        self.learned_config["confidence"] = self.confidence
        self.learned_config["regions_model_input_type"] = self.regions_model_input_type

        if self.regions_model_type == "multi_region":
            self.learned_config["leaf_values"] = self.regions_model.leaf_values
            base_onnx_model = onnx.load_model_from_string(self.base_model)
            # avoid output name conflicts between different models
            base_and_region_model = onnx.compose.add_prefix(
                base_onnx_model,
                prefix="base_model_",
                rename_nodes=False,
                rename_edges=False,
                rename_inputs=False,
                rename_outputs=True,
                rename_initializers=False,
                rename_value_infos=False,
                rename_functions=False,
                inplace=False,
            )

            # add identity node
            id_node = onnx.helper.make_node(
                "Identity",
                inputs=[base_and_region_model.graph.input[0].name],
                outputs=["x_orig"],
            )
            base_and_region_model.graph.node.insert(
                len(base_and_region_model.graph.node), id_node
            )

            # add identity node output
            id_output_var = copy.deepcopy(base_and_region_model.graph.input[0])
            id_output_var.name = "x_orig"
            base_and_region_model.graph.output.insert(0, id_output_var)

            # find opset ai.onnx and ai.onnx.ml in base model
            base_opset_ai_onnx = None
            base_opset_ai_onnx_indx = None
            base_opset_ai_onnx_ml = None
            base_opset_ai_onnx_ml_indx = None
            base_opset_list = (
                []
            )  # what if it is not present in base model? - should be added to the model
            for field in base_and_region_model.ListFields():
                if field[0].name == "opset_import":
                    base_opset_list = field[1]
                    for i, operatorIdProto in enumerate(field[1]):
                        if (not operatorIdProto.HasField("domain")) or (
                            operatorIdProto.domain in ["", "ai.onnx"]
                        ):
                            base_opset_ai_onnx = operatorIdProto
                            base_opset_ai_onnx_indx = i
                        elif operatorIdProto.domain == "ai.onnx.ml":
                            base_opset_ai_onnx_ml = operatorIdProto
                            base_opset_ai_onnx_ml_indx = i

            if base_opset_ai_onnx is None:
                base_opset_ai_onnx = onnx.onnx_ml_pb2.OperatorSetIdProto(
                    version=self.opset_version, domain="ai.onnx"
                )
                base_opset_list.append(base_opset_ai_onnx)

            # cp region model
            region_model = to_onnx(
                self.regions_model.region_model,
                X_input[:1].astype(np.float32),
                target_opset=base_opset_ai_onnx.version,
            )
            region_model_str = region_model.SerializeToString()
            self.learned_config["region_model"] = base64.b64encode(
                region_model_str
            ).decode("ascii")

            # verify versions
            # cp region opset
            region_opset_ai_onnx = None
            region_opset_ai_onnx_indx = None
            region_opset_ai_onnx_ml = None
            region_opset_ai_onnx_ml_indx = None
            for field in region_model.ListFields():
                if field[0].name == "opset_import":
                    for i, operatorIdProto in enumerate(field[1]):
                        if (not operatorIdProto.HasField("domain")) or (
                            operatorIdProto.domain in ["", "ai.onnx"]
                        ):
                            region_opset_ai_onnx = operatorIdProto
                            region_opset_ai_onnx_indx = i
                        elif operatorIdProto.domain == "ai.onnx.ml":
                            region_opset_ai_onnx_ml = operatorIdProto
                            region_opset_ai_onnx_ml_indx = i

            if (base_opset_ai_onnx is not None) and (region_opset_ai_onnx is not None):
                if base_opset_ai_onnx.version != region_opset_ai_onnx.version:
                    raise Exception(
                        "base model opset version for ai.onnx {} is different from region opset version {}".format(
                            base_opset_ai_onnx.version, region_opset_ai_onnx.version
                        )
                    )

            if (base_opset_ai_onnx_ml is not None) and (
                region_opset_ai_onnx_ml is not None
            ):
                if base_opset_ai_onnx_ml.version != region_opset_ai_onnx_ml.version:
                    raise Exception(
                        "base model opset version for ai.onnx.ml {} is different from region opset version {}".format(
                            base_opset_ai_onnx_ml.version,
                            region_opset_ai_onnx_ml.version,
                        )
                    )

            if base_opset_ai_onnx_ml is None:
                base_opset_list.append(region_opset_ai_onnx_ml)

            # add region model to base model pipeline
            base_and_region_model.graph.node.insert(
                len(base_and_region_model.graph.node), region_model.graph.node[0]
            )

            # add region model output
            base_and_region_model.graph.output.insert(
                len(base_and_region_model.graph.output), region_model.graph.output[0]
            )

            # graph name
            base_and_region_model.graph.name = (
                "Modified_ONNX_Pipeline_with_Region_Model"
            )

            # mode metadata
            base_and_region_model.producer_name = "Region_Uncertainty_Estimator"
            if base_and_region_model.HasField("model_version"):
                base_and_region_model.model_version = (
                    base_and_region_model.model_version + 1
                )
            else:
                base_and_region_model.model_version = 1

            # initialize inference class for multi-region
            torch_cp_predictor = CPInferenceMultiRegion(
                conformity_scores_list=self.regions_stats,
                leaf_values=self.regions_model.leaf_values,
                quantile=self.confidence / 100,
            )
            base_and_region_model_sess = rt.InferenceSession(
                base_and_region_model.SerializeToString()
            )
            # get sample data points for torch inference
            upto_tree_model_output = base_and_region_model_sess.run(
                None, {"X": X_input[0:1].astype(np.float32)}
            )

            cp_inputs = {  # x_orig, base_prediction, region_prediction
                "x_orig": torch.from_numpy(upto_tree_model_output[0]).float(),
                "base_prediction": torch.from_numpy(upto_tree_model_output[1]).float(),
                "region_prediction": torch.from_numpy(
                    upto_tree_model_output[2]
                ).float(),
            }

            torch_input_names = ["x_orig", "base_prediction", "region_prediction"]

            torch_output_names = [
                "input_x",
                "prediction",
                "region_model_prediction",
                "exp_lower_pred_endpoint",
                "exp_upper_pred_endpoint",
            ]

            cp_model_bytes = io.BytesIO()
            torch.onnx.export(
                torch_cp_predictor,
                cp_inputs,
                cp_model_bytes,
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=self.opset_version,
                input_names=torch_input_names,  # the model's input names
                output_names=torch_output_names,  # the model's output names
                dynamic_axes={
                    torch_input_names[0]: {0: "batch_size"},  # variable length axes
                    torch_input_names[1]: {0: "batch_size"},
                    torch_input_names[2]: {0: "batch_size"},
                    torch_output_names[0]: {0: "batch_size"},
                    torch_output_names[1]: {0: "batch_size"},
                    torch_output_names[2]: {0: "batch_size"},
                    torch_output_names[3]: {0: "batch_size"},
                    torch_output_names[4]: {0: "batch_size"},
                },
            )
            cp_onnx_model = onnx.load_model_from_string(cp_model_bytes.getvalue())

            combined_model = merge_models(
                base_and_region_model,
                cp_onnx_model,
                io_map=[
                    (
                        base_and_region_model.graph.output[0].name,
                        torch_input_names[0],
                    ),
                    (
                        base_and_region_model.graph.output[1].name,
                        torch_input_names[1],
                    ),
                    (
                        base_and_region_model.graph.output[2].name,
                        torch_input_names[2],
                    ),
                ],
            )
            self.learned_config["combined_model"] = combined_model.SerializeToString()

        else:
            torch_cp_predictor = CPInferenceSingleRegion(
                conformity_scores=np.array(self.regions_stats[0]),
                quantile=self.confidence / 100,
            )
            cp_input = torch.from_numpy(y_pred.reshape(-1, 1)[:1]).float()

            cp_model_bytes = io.BytesIO()
            torch.onnx.export(
                torch_cp_predictor,
                cp_input,
                cp_model_bytes,
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=self.opset_version,
                input_names=[torch_input_name],  # the model's input names
                output_names=torch_output_names,  # the model's output names
                dynamic_axes={
                    torch_input_name: {0: "batch_size"},  # variable length axes
                    torch_output_names[0]: {0: "batch_size"},
                    torch_output_names[1]: {0: "batch_size"},
                    torch_output_names[2]: {0: "batch_size"},
                },
            )
            cp_onnx_model = onnx.load_model_from_string(cp_model_bytes.getvalue())
            base_onnx_model = onnx.load_model_from_string(self.base_model)
            combined_model = merge_models(
                base_onnx_model,
                cp_onnx_model,
                io_map=[
                    (
                        base_onnx_model.graph.output[0].name,
                        torch_input_name,
                    )
                ],  # Link i/o
            )

            self.learned_config["combined_model"] = combined_model.SerializeToString()

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

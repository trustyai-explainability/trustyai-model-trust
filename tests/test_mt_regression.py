import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from model_trust.base.region_identification.region_identification_models import RegionQuantileTreeIdentification, SingleRegion
from model_trust.regression.region_uncertainty_estimation import (
    RegionUncertaintyEstimator,
)
from model_trust.utils.synthetic_data import get_linear_2_region


from scipy.stats import norm

seed = 42
nsamples = 3000
quantile = 0.9


sigma_0 = 1
sigma_1 = 4

np.random.seed(seed)
data_dic = get_linear_2_region(seed=seed,nsamples=nsamples,quantile=quantile,sigma_0=sigma_0,sigma_1=sigma_1)
pd_data = data_dic['data']
features = data_dic['features']
target = data_dic['target']


X = pd_data[features].values.flatten()
Y = pd_data[target].values

''' GT for reference'''
pi_gt_low = pd_data['PI_GT_LOW'].values
pi_gt_high = pd_data['PI_GT_HIGH'].values
Y_mean = pd_data['Y_mean'].values
coverage_empirical = np.mean((Y <=pi_gt_high)& (Y >=pi_gt_low))
x_data = pd_data[features].values
y_data = pd_data[target].values
test_ratio = 0.2
cal_ratio = 0.2



print(x_data.shape, y_data.shape)

X_tr, X_test, Y_tr, Y_test = train_test_split(
    x_data, y_data, test_size=test_ratio, random_state=42
)

X_train, X_cal, Y_train, Y_cal = train_test_split(
    X_tr, Y_tr, test_size=cal_ratio, random_state=42
)

Y_train = Y_train.squeeze()
Y_test = Y_test.squeeze()
Y_cal = Y_cal.squeeze()

print("X_train, y_train : ", X_train.shape, Y_train.shape)
print("X_cal, Y_cal : ", X_cal.shape, Y_cal.shape)
print("X_test, Y_test : ", X_test.shape, Y_test.shape)

base_model = LinearRegression().fit(X_train, Y_train)
print("Train/Test R2 scores")
print(base_model.score(X_train, Y_train), base_model.score(X_test, Y_test))

""" 
REGION MODEL
"""

model_selection_metric = "coverage_ratio"
model_selection_stat = "min"
min_group_size = 20

region_model = RegionQuantileTreeIdentification(
    surrogate_model="lgbm",
    model_selection_metric=model_selection_metric,
    model_selection_stat=model_selection_stat,
    min_group_size=min_group_size,
    quantile=quantile,
)

""" 
CONFORMAL PREDICTION MODEL
"""

cp_config_input = {}
cp_config_input["predictive_model"] = base_model
cp_config_input["regions_model"] = region_model
cp_config_input["method"] = "prefit"
cp_config_input["confidence"] = quantile * 100


umodel = RegionUncertaintyEstimator(**cp_config_input)


""" 
BASELINE: SINGLE REGION MODEL CONFORMAL PREDICTION MODEL
"""


cp_config_input["regions_model"] = SingleRegion()
umodel_base = RegionUncertaintyEstimator(**cp_config_input)

if umodel.method == "prefit":
    umodel = umodel.fit(X_cal, Y_cal)
else:
    umodel = umodel.fit(
        np.concatenate([X_train, X_cal], axis=0),
        np.concatenate(
            [Y_train[:, np.newaxis], Y_cal[:, np.newaxis]], axis=0
        ).squeeze(),
    )


Y_pred_regioncp = umodel.predict_interval(np.sort(X_test))
Y_pred_scp = umodel_base.predict_interval(np.sort(X_test))

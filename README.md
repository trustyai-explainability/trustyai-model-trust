# trustyai-model-trust
This repository offers split conformal prediction to estimate the uncertainty of an arbitrary machine learning (ML) model. 
​
We provide post-hoc conformal prediction (CP) wrappers that turn the prediction of a ML regression model into a prediction interval (or set) that contains the target variable within a user specified probability on average (also known as coverage/validity, sometimes denoted as confidence). The CP algorithm only requires access to the prediction function of the ML model, a calibration dataset that was not used to train the ML model, and the probability/coverage/confidence for the prediction interval. The calibration dataset should reflect the type of data that the model will see when deployed. This CP algorithm is enhanced standard split conformal prediction wrappers with a region discovery approach that identifies groups in the input space where the conformity scores are significantly different. This allows us to better characterize heteroscedastic uncertainty and improve the coverage of the prediction intervals on the discovered groups in addition to its average guarantee. The discovered groups are regions in the input space where the error of the model is significantly different, and consequently the prediction interval/set sizes differ (i.e., uncertainty or precision of the model prediction). 
​

## Current Conformal Prediction Methods
​
### Post-hoc Conformal Prediction for Regression
We currently support post-hoc conformal prediction wrappers for ML regression models that predict a real-valued variable from multivariate input. The current implementation uses the absolute error as conformity scores producing prediction intervals centered at the prediction of the provided regression model.
​
### Single/Multi-Region Conformal Prediction
We support single region (standard) and multi-region split conformal prediction based on decision trees. The latter indicates that the error regions are learned by a decision tree clustering approach. The identified groups are inputs to the conformal prediction method which enables adjustments of the prediction intervals/sets based on the uncertainty of the discovered regions. 
​

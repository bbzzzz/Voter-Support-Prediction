# Voter-Support-Prediction
This is a project I did for DC based analytics company BlueLabs.

BlueLabs is an analytics, data, and technology company formed by senior members of the Obama for America analytics team. 

In this project, I built a model that predicts voter's probability of supporting the Democratic candidate versus the Republican candidate for BlueLabs.

The main work I did in the process of building the predictive model:
- Missing data imputation
- Feature extraction and feature engineering
- Feature selection
- Parameter tuning using grid search
- Cross validation
- Optimal threshold value selection

One machine learning model is trained and tested:
- Gradient Boosted Tree using xgboost

Model performance is measured by AUC score. Final model AUC: 0.921.

Upon the results of prediction model, several suggestions are made to end users at different levels, which can be found in the summary report.

# This code is used for BlueLabs modeling simulation.
# Author: Bohan Zhang

library(readr)         # read data
library(lubridate)     # extract date information
library(caret)         # data preprocessing and model parameter tuning
library(corrplot)      # correlation plot
library(xgboost)
library(randomForest)
library(pROC)
library(dplyr)
library(tidyr)
library(Ckmeans.1d.dp) # plot xgb importance
library(ROCR)

# Please unzip the data an folder named input under the working directory
cat("reading the train and test data\n")
train <- read.csv(file="./input/bluelabs_modeling_data.csv",head=TRUE,sep=",")
test  <- read.csv(file="./input/bluelabs_scoring_data.csv",head=TRUE,sep=",")

# Delete a column (persuasion_pct) w/ all NAs
train <- train[,-10]      
test <- test[,-10]

# Dealing w/ some inconsistancy in train set
# use mode to fill na
train$state_region[train$state_region==""] <- names(sort(-table(train$state_region)))[1]
train$state_region <- train$state_region[drop=TRUE]

# Dealing w/ some inconsistancy in test set
# sex
test$sex[test$sex == ""] <- names(sort(-table(test$sex)))[1]
test$sex[test$sex == "U"] <- names(sort(-table(test$sex)))[1]
test$sex <- test$sex[drop=TRUE]

# state_region
test$state_region[test$state_region == ""] <- names(sort(-table(test$state_region)))[1]
test$state_region <- test$state_region[drop=TRUE]

# ethnicity
test$ethnicity[test$ethnicity == ""] <- names(sort(-table(test$ethnicity)))[1]
test$ethnicity <- test$ethnicity[drop=TRUE]

############################################################################## 
# Feature Engineering
# Seperating year, month, day and day of week from the registration date feature
train$earliest_reg_date <- as.Date(train$earliest_reg_date)
train$year <- year(train$earliest_reg_date)
train$month <- month(train$earliest_reg_date)
train$day <- day(train$earliest_reg_date)
train$weekday <- wday(train$earliest_reg_date)

test$earliest_reg_date <- as.Date(test$earliest_reg_date)
test$year <- year(test$earliest_reg_date)
test$month <- month(test$earliest_reg_date)
test$day <- day(test$earliest_reg_date)
test$weekday <- wday(test$earliest_reg_date)

# Removing the date column (since elements are extracted)
train <- train[,-12]
test <- test[,-12]

# Ratio/Difference - Conservative to Libearl
train$con_lib_ratio <- train$conservative/train$liberal
train$con_lib_dff <- train$conservative - train$liberal

test$con_lib_ratio <- test$conservative/test$liberal
test$con_lib_dff <- test$conservative - test$liberal

# Low income: income < cen_medianincome
train$low_income <- ifelse(train$income<train$cen_medianincome,1,0)
test$low_income <- ifelse(test$income<test$cen_medianincome,1,0)

# Total number of vote from 2004 - 2012
train$total_vote_r <- train$vote_p2004party_r + train$vote_p2006party_r +
  train$vote_p2008party_r + train$vote_p2009party_r + 
  train$vote_p2010party_r + train$vote_p2011party_r +
  train$vote_p2012party_r

train$total_vote_d <- train$vote_p2004party_d + train$vote_p2006party_d +
  train$vote_p2008party_d + train$vote_p2009party_d + 
  train$vote_p2010party_d + train$vote_p2011party_d +
  train$vote_p2012party_d

test$total_vote_r <- test$vote_p2004party_r + test$vote_p2006party_r +
  test$vote_p2008party_r + test$vote_p2009party_r + 
  test$vote_p2010party_r + test$vote_p2011party_r +
  test$vote_p2012party_r

test$total_vote_d <- test$vote_p2004party_d + test$vote_p2006party_d +
  test$vote_p2008party_d + test$vote_p2009party_d + 
  test$vote_p2010party_d + test$vote_p2011party_d +
  test$vote_p2012party_d

# Golden Future: difference of two highly correlated features:
#       = dem_performance_pct - random_var
train$golden_feature_1 <- train$dem_performance_pct - train$random_var
train$golden_feature_2 <- train$dem_performance_pct/train$random_var

test$golden_feature_1 <- test$dem_performance_pct - test$random_var
test$golden_feature_2 <- test$dem_performance_pct/test$random_var

# Extract all feature names, excluding support and id
feature.names <- names(train)[-c(56,60)]

# Define a function to create Dummy Variables for categorical variables: 
# sex, state_region, ethnicity
create_dummy <- function(f,data){
  if (class(data[[f]])=="factor") {
    n <- length(data[[f]])
    data.fac <- data.frame(x = data[[f]],y = 1:n)
    dummy_matrix <- model.matrix(y~x,data.fac)[,-1]
    dummy_df = data.frame(dummy_matrix)
    colnames(dummy_df) <- paste(f,"_",levels(data[[f]])[-1],sep="")
  
  return(dummy_df)
  }
}

for (f in feature.names) {
  if (class(train[[f]])=="factor") {
    train <- cbind(train,create_dummy(f,train))
    test <- cbind(test,create_dummy(f,test)) 
  }
}

# Convert categorical featrues to numeric
for (f in feature.names) {
  if (class(train[[f]])=="factor") {
    train[[f]] <- as.integer(train[[f]])
    test[[f]]  <- as.integer(test[[f]])
  }
}

# Update the feature name list
feature.names <- names(train)[-c(56,60)]
##############################################################################
# Missing value imputation
# Fiil all na with -1 as indicator of missing value
train_imputed_neg1 <- train
test_imputed_neg1 <- test

train_imputed_neg1[is.na(train)] <- -1
test_imputed_neg1[is.na(test)] <- -1

# KNN impution using caret preprocessing function
zero_var_cols <- nearZeroVar(train)
zero_var_names <- names(train)[zero_var_cols]

train_filter <- names(train) %in% c("id","support",zero_var_names)
train_knn <- train[!train_filter]

test_filter <- names(test) %in% c("id","support",zero_var_names)
test_knn <- test[!test_filter]

pp <- preProcess(train_knn, method = "knnImpute")
train_imputed_knn <- predict(pp,train_knn)
test_imputed_knn <- predict(pp,test_knn)

train_imputed_knn <- cbind(train_imputed_knn,train[train_filter])
test_imputed_knn <- cbind(test_imputed_knn,test[test_filter])

# Fill the else with -1
train_imputed_knn[is.na(train_imputed_knn)] <- -1
test_imputed_knn[is.na(test_imputed_knn)] <- -1

############################################################################## 
# We use 10-fold cv and auc to measure the model
# A function to perform xgboost on 10-fold cross validation
# and output AUC score
xgb <- function(tra){
  
  set.seed(1778)
  
  h<-sample(nrow(tra),nrow(tra)*0.1)
  
  dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$support[h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$support[-h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$support)
  
  watchlist<-list(val=dval,train=dtrain) # Build a watchlist for early stop
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = 0.01,       # 0.06, #0.01,
                  max_depth           = 4,          # changed from default of 8
                  subsample           = 1,          # 0.7
                  colsample_bytree    = 0.7,        # 0.7
                  gamma               = 0.1
  )
  
  cv <- xgb.cv( params           = param, 
                data             = dtrain, 
                nrounds          = 1500, 
                nfold            = 10,       # number of folds in K-fold
                prediction       = TRUE,     # return the prediction using the final model 
                showsd           = TRUE,     # standard deviation of loss across folds
                stratified       = FALSE,    # sample is unbalanced; use stratified sampling
                verbose          = TRUE,
                print.every.n    = 100,
                early.stop.round = 100       # if performance don't improve, stop
  )
    
  print(paste("Best test AUC:",max(cv$dt$test.auc.mean)))
  
  return(cv)
}

xgb_plot <- function(cv){
  # plot the AUC for the training and testing samples
  cv$dt %>%
    select(-contains("std")) %>%
    mutate(IterationNum = 1:n()) %>%
    gather(TestOrTrain, AUC, -IterationNum) %>%
    ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
    geom_line() + 
    theme_bw()
}

# Perfrom Random Forest model with cross validation 
# and output average AUC score
rf <- function(tra,ntree){
  set.seed(1778)
  
  cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 10, 
                          summaryFunction = twoClassSummary,
                          classProbs = TRUE,
                          allowParallel=TRUE)
  
  tra$support <- as.factor(train$support)
  levels(tra$support) <- c('Yes','No')
  
  rf_tune <- train(support~.,
                   data=tra,
                   method="rf",
                   trControl=cv.ctrl,
                   tuneGrid=data.frame(mtry=c(15)),
                   verbose=T,
                   ntree=ntree,
                   metric="ROC"
  )
  print(rf_tune)
  return(rf_tune)
}

############################################################################## 
# Model 1: orignial feature set
features_1 <- feature.names[1:92]
train_1_neg <- train_imputed_neg1[,features_1]
train_1_knn <- train_imputed_knn[,features_1]
xgb_1_neg <- xgb(train_1_neg)
xgb_1_knn <- xgb(train_1_knn)
# xgboost: 0.919689, neg impute
# xgboost: 0.919945, knn impute

rf_1_neg <- rf(train_1_neg,1000)
rf_1_knn <- rf(train_1_knn,1000)
# rf: 0.9115394, neg impute, mtry 15, ntree 1000
# rf: 0.911913, knn impute, mtry 15, ntree 1000

# Model 2: (original + engineered) feature
features_2 <- feature.names
train_2_neg <- train_imputed_neg1[,features_2]
train_2_knn <- train_imputed_knn[,features_2]
xgb_2_neg <- xgb(train_2_neg)
xgb_2_knn <- xgb(train_2_knn)
# xgboost: 0.919747, neg impute
# xgboost: 0.920042, knn impute

rf_2_neg <-  rf(train_2_neg,1000)
rf_2_knn <- rf(train_2_knn,1000)
# rf: 0.91354, neg impute, mtry 15, ntree 1000
# rf: 0.9141558, knn impute, mtry 15, ntree 1000

# Model 3: (original + new - zero_var) feature
#          Take out all features with zero or near zero variance
#          use caret package nearZeroVar() function
zero_var_cols <- nearZeroVar(train)
zero_var_names <- names(train)[zero_var_cols]
features_3 <- feature.names[!(feature.names %in% zero_var_names)]
train_3_neg <- train_imputed_neg1[,features_3]
train_3_knn <- train_imputed_knn[,features_3]
xgb_3_neg <- xgb(train_3_neg)
xgb_3_knn <- xgb(train_3_knn)
# xgboost: 0.920116, neg impute
# xgboost: 0.919756, knn impute

rf_3_neg <- rf(train_3_neg,1000)
rf_3_knn <- rf(train_3_knn,1000)
# rf: 0.912352, neg impute, mtry 15, ntree 1000
# rf: 0.912352, neg impute, mtry 15, ntree 1000
############################################################################## 
# At this step, we want to do feature selection based on feature importance
# Here is an function to extact feature imporance from xgb model
xgb_importance <- function(tra){
  set.seed(1778)
  
  # Sample a small part from the data set as validation set
  h<-sample(nrow(tra),nrow(tra)*0.1)
  
  # Build train set and validation set
  dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$support[h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$support[-h])
  
  # Build a watchlist for early stop using the validation set
  watchlist<-list(val=dval,train=dtrain) 
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = 0.01,       # Step size
                  max_depth           = 4,          
                  subsample           = 1,          
                  colsample_bytree    = 0.7,        
                  gamma               = 0.1
  )
  
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 1500, 
                      verbose             = 1,  
                      early.stop.round    = 100,
                      print.every.n       = 100,
                      watchlist           = watchlist
  )
  
  # Get feature importance from the trained model
  importance <- xgb.importance(names(tra), model = clf)
  
  return(importance)
}

# train the model using data from last step and get feature importance
importance_neg <- xgb_importance(train_3_neg)
xgb.plot.importance(importance_neg[1:10,]) # plot top 10 important features
importance_neg$Feature   # a sorted list of features according to importance 
# least contributed features:
# "state_region_Region 5"     "state_region_Region 2"     "bookmusc_1"               
# "smarstat_s"  "state_region_Region 6"  
# "vote_g2008"  "state_region_Region 4"

importance_knn <- xgb_importance(train_3_knn)
xgb.plot.importance(importance_knn[1:10,])
importance_knn$Feature   # a sorted list of features according to importance 
# least contributed features:
# "electrnc_1"  "cen_ruralpcnt"  "cen_urbanpcnt"  "smarstat_s"  "vote_g2008"     
# "bookmusc_1" "state_region_Region 4" "state_region_Region 6"

#setdiff(features_3,importance_neg$Feature)
#setdiff(features_3,importance_knn$Feature)

# Considering the two parts, we got a short list of features we want to remove:
features_to_remove <- c("smarstat_s","bookmusc_1","vote_g2008",
                        "state_region_Region 2","state_region_Region 3",
                        "state_region_Region 4","state_region_Region 5",
                        "state_region_Region 6")
features_4 <- features_3[!(features_3 %in% features_to_remove)]
train_4_neg <- train_imputed_neg1[,features_4]
train_4_knn <- train_imputed_knn[,features_4]
xgb_4_neg <- xgb(train_4_neg)
xgb_4_knn <- xgb(train_4_knn)
# xgboost: 0.920222, neg impute
# xgboost: 0.919887, knn impute

###############################################
# Fine tune hyperparameters of xgboost with grid search
xgb_tune <- function(tra){
  set.seed(1778)
  
  cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 5, 
                          summaryFunction = twoClassSummary,
                          classProbs = TRUE,
                          allowParallel=TRUE)
  
  xgb.grid <- expand.grid(nrounds = c(1500),
                          max_depth = c(2,3,4),
                          eta = c(0.005,0.01),
                          gamma = c(0.01,0.1),
                          colsample_bytree = c(0.5,0.7),
                          min_child_weight = c(3,5)
  )
  
  tra$support <- as.factor(train$support)
  levels(tra$support) <- c('Yes','No')
  
  xgb_tune <- train(support~.,
                    data=tra,
                    method="xgbTree",
                    trControl=cv.ctrl,
                    tuneGrid=xgb.grid,
                    verbose=T,
                    metric="ROC"
  )
  
  return(xgb_tune)
}

xgb_tune_knn <- xgb_tune(train_2_knn)
# 0.010   3   0.01    0.5   3   1500    0.9209165
xgb_tune_neg <- xgb_tune(train_4_neg)
# 0.005   4   0.01    0.5   5   1500    0.9203164

###############################################
# Find feature importance from the fine-tuned models
# KNN Imputed data sets w/ all features (feature set 2)
set.seed(1778)
  
# Sample a small part from the data set as validation set
h<-sample(nrow(train_2_knn),nrow(tra)*0.1)
  
# Build train set and validation set
dval<-xgb.DMatrix(data=data.matrix(train_2_knn[h,]),label=train$support[h])
dtrain<-xgb.DMatrix(data=data.matrix(train_2_knn[-h,]),label=train$support[-h])
  
# Build a watchlist for early stop using the validation set
watchlist<-list(val=dval,train=dtrain) 
  
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.005,       # Step size
                max_depth           = 3,          
                subsample           = 1,          
                colsample_bytree    = 0.5,
                min_child_weight    = 3,
                gamma               = 0.01
                )
  
clf_knn <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1500, 
                    verbose             = 1,  
                    early.stop.round    = 200,
                    print.every.n       = 100,
                    watchlist           = watchlist
                    )
  
# Get feature importance from the trained model
importance_tune_knn <- xgb.importance(names(train_2_knn), model = clf_knn)
xgb.plot.importance(importance_tune_knn[1:10,])

###############################################
# Find feature importance from the fine-tuned models
# Neg_1 Imputed data sets w/ filtered features (feature set 4):
set.seed(1778)

# Sample a small part from the data set as validation set
h<-sample(nrow(train_4_neg),nrow(tra)*0.1)

# Build train set and validation set
dval<-xgb.DMatrix(data=data.matrix(train_4_neg[h,]),label=train$support[h])
dtrain<-xgb.DMatrix(data=data.matrix(train_4_neg[-h,]),label=train$support[-h])

# Build a watchlist for early stop using the validation set
watchlist<-list(val=dval,train=dtrain) 

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.005,       # Step size
                max_depth           = 4,          
                subsample           = 1,          
                colsample_bytree    = 0.5, 
                min_child_weight    = 5
                gamma               = 0.01
)

clf_neg <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1500, 
                    verbose             = 1,  
                    early.stop.round    = 200,
                    print.every.n       = 100,
                    watchlist           = watchlist
)

# Get feature importance from the trained model
importance_tune_neg <- xgb.importance(names(train_4_neg), model = clf_neg)
xgb.plot.importance(importance_tune_neg[1:10,])

###############################################
# Output the probablibities
pred_neg <- predict(clf_neg, data.matrix(test_imputed_neg1[,features_4]))
pred_knn <- predict(clf_knn, data.matrix(test_imputed_knn[,features_2]))

# The final prediction is the average of prediction by two models
pred <- (pred_neg + pred_knn)/2

submission <- data.frame(id=test$id, support=pred)
cat("saving the submission file\n")
write_csv(submission, "./output/pred_prob.csv")

###############################################
# Find the optimal threshold
# The optimal threshold should optimize the sum of sensitivity and specificity
opt_threshold <- function(predict, response) {
  r <- pROC::roc(response, predict)
  r$thresholds[which.max(r$sensitivities + r$specificities)]
}

# The final prediction is the average of prediction by two models
pred_val <- (xgb_2_knn$pred + xgb_4_neg$pred)/2

cat("Optimal Threshold is\n") 
opt_threshold(pred_val,train$support)     # 0.5098609

##############################################

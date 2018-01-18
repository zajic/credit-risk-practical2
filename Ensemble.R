#################################################################################################
# This file is just for reference, it came with some bugs so it cannot be run.
################################################################################################

library(plyr)
  # Full
  creditset$GOOD <- mapvalues(creditset$GOOD,from = c("1", "0"), to = c("GOOD", "BAD"))
  
  revalue(creditset$GOOD, c("1"="GOOD", "0"="BAD"))
  
  creditset$CA_01 <- mapvalues(creditset$CA_01,
                               from = c("1", "2", "3", "4", "5"),
                               to = c("one", "two", "three", "four", "five"))
  
  revalue(creditset$CA_01, c("1"="one", "2"="two", "3"="three", "4"="four", "5"="five"))
  
  # Train
  train_set$GOOD <- mapvalues(train_set$GOOD,from = c("1", "0"), to = c("GOOD", "BAD"))
  
  revalue(train_set$GOOD, c("1"="GOOD", "0"="BAD"))
  
  train_set$CA_01 <- mapvalues(train_set$CA_01,
                               from = c("1", "2", "3", "4", "5"),
                               to = c("one", "two", "three", "four", "five"))
  
  revalue(train_set$CA_01, c("1"="one", "2"="two", "3"="three", "4"="four", "5"="five"))
  
  # Test
  test_response <- mapvalues(test_response,from = c("1", "0"), to = c("GOOD", "BAD"))
  
  revalue(test_response, c("1"="GOOD", "0"="BAD"))
  
  test_predictor$CA_01 <- mapvalues(test_predictor$CA_01,
                                    from = c("1", "2", "3", "4", "5"),
                                    to = c("one", "two", "three", "four", "five"))
  
  revalue(test_predictor$CA_01, c("1"="one", "2"="two", "3"="three", "4"="four", "5"="five"))
  
  
  # Known bug caretEnsemble naming problem
  
  library("caretEnsemble")
  library("mlbench")
  library("randomForest")
  library("nnet")
  model_list_big <- caretList(
    GOOD ~ ., data=train_set,
    trControl=trainControl(summaryFunction=twoClassSummary,classProbs=TRUE,savePrediction="final",verboseIter = TRUE),
    methodList=c("glm", "rpart"),
    metric="Accuracy",
    tuneList=list(
      .aa=caretModelSpec(method="rf", tuneGrid=expand.grid(mtry = 17)),
      .bb=caretModelSpec(method="rf", tuneGrid=expand.grid(mtry = 23)),
      .cc=caretModelSpec(method="rf", tuneGrid=expand.grid(mtry = 5)),
      .dd=caretModelSpec(method="nnet", tuneGrid=expand.grid(size=18,decay=0.05),maxit=1500),
      .ee=caretModelSpec(method="nnet", tuneGrid=expand.grid(size=15,decay=0.05),maxit=1500),
      .ff=caretModelSpec(method="svmLinear", tuneGrid=expand.grid(C=1))
    )
  )
  # as.factor(occ_code) + as.factor(time_emp) + as.factor(res_indicator) +
  # cust_age + as.factor(CA_01) + CA_02 + as.factor(CA_03) + ER_01 + ER_02 +
  #  as.factor(S_01) + as.factor(S_02) + disp_income + as.factor(I_01) +
  #  I_02 + I_03 + I_04 + D_01 + D_02 + I_05 + I_06 + P_01
  
  set.seed(476457)
  
  rf_ensemble <- caretStack(
    model_list_big,
    method="glm",
    metric="Accuracy",
    trControl=trainControl(method="cv", number=5, summaryFunction=twoClassSummary,classProbs=TRUE,savePrediction="final",verboseIter = TRUE)
  )
  
  summary(rf_ensemble)
  
  ens_preds <- predict(rf_ensemble, newdata=test_predictor, type="prob")
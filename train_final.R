##############################################################################################
#
# Init 
#
##############################################################################################

library(caret)
library(caTools)

library(foreach)
library(doParallel)
registerDoParallel(cores=4)

library(plyr)

set.seed(34576345)

NTREE = 1500           # For most cases/tests, 500 to 1000 really is enough.
MTRY_RANGE = 1:30
# NOTE: I suspect the randomForest package does automatic categorical var encoding,
#       after which there may be more than the original 22 variables, so perhaps
#       mtry could have a wider range than 1:22

levelsFromNumToText <- function(dataset) {
  if (!is.null(dataset$GOOD)) {
    dataset$GOOD <- mapvalues(dataset$GOOD,
                              from = c("1", "0"),
                              to = c("GOOD", "BAD"))
  }
  
  # revalue(dataset$GOOD, c("1"="GOOD", "0"="BAD"))
  
  dataset$CA_01 <- mapvalues(dataset$CA_01,
                             from = c("1", "2", "3", "4", "5"),
                             to = c("one", "two", "three", "four", "five"))
  
  # revalue(dataset$CA_01, c("1"="one", "2"="two", "3"="three", "4"="four", "5"="five"))
  
  return(dataset)
}

fitModel <- function(trainset_path, testset_path) {
  ##############################################################################################
  #
  # Load data and some more pre-processing of data 
  #
  ##############################################################################################
  
  creditset <- read.csv(trainset_path)
  creditset$app_id <- NULL
  creditset$GOOD  <- as.factor(creditset$GOOD)
  creditset$CA_01 <- as.factor(creditset$CA_01)
  creditset <- levelsFromNumToText(creditset)
  
  testset <- read.csv(testset_path)
  testset$app_id <- NULL
  testset$GOOD  <- as.factor(testset$GOOD)
  testset$CA_01 <- as.factor(testset$CA_01)
  testset <- levelsFromNumToText(testset)
  
  test_response  <- testset[, which(colnames(testset)=="GOOD")]        # test set response
  test_predictor <- testset[,-which(colnames(testset)=="GOOD")]        # test set predictors
  
  ########################
  # Fit the model
  ########################
  
  rfGrid <- expand.grid(mtry = MTRY_RANGE)
  
  rfControl <- trainControl(method = "oob",
                            search = "random",
                            verboseIter = TRUE,
                            returnData = TRUE,
                            returnResamp = "final",
                            savePredictions = FALSE,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary,
                            selectionFunction = "best",
                            preProcOptions = NULL,
                            sampling = NULL,
                            index = NULL,
                            indexOut = NULL,
                            indexFinal = NULL,
                            timingSamps = 0,
                            trim = FALSE,
                            allowParallel = TRUE)
  
  rf_model <- train(GOOD ~ .,
                    data = creditset,
                    method = "rf",
                    tuneGrid = rfGrid,
                    trControl = rfControl,
                    metric = "Accuracy",
                    ntree = NTREE)
  
  ############
  # Evaluate
  ############
  
  # the randomForest's confusion matrix estimate based on OOB error estimate 
  rf_model$finalModel$confusion
  
  # does it agree with test set confusion matrix?
  predicted <- predict(rf_model,newdata = test_predictor)
  cm <- confusionMatrix(predicted,test_response,positive="GOOD")
  
  results <- list("caretModel" = rf_model,
                  "testsetConfusion" = cm,
                  "testSet" = testset,
                  "trainSet" = creditset,
                  "trainsetPath" = trainset_path,
                  "testsetPath" = testset_path)
  
  return(results)
}

result <- fitModel("dataset_imputed_final_train.csv", "dataset_imputed_final_test.csv")

save.image("final.RData")
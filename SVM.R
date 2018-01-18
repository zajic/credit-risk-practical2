##############################################################################################

library(caret)
library(caTools)
library(foreach)
library(doParallel)
registerDoParallel(cores=4)

set.seed(34576345)

##############################################################################################
#
# Load data and some more pre-processing of data 
#
##############################################################################################

creditset <- read.csv("csv_files/dataset_imputed_known.csv")
creditset$X <- NULL
creditset$GOOD  <- as.factor(creditset$GOOD)
creditset$CA_01 <- as.factor(creditset$CA_01)

########################
# Split train test set
########################

# Using sample.split from caTools
Y = creditset$GOOD
msk = sample.split(Y,SplitRatio=0.7)
train_set = creditset[msk,]
test_set  = creditset[!msk,]
test_response  <- test_set[, 1]        # test set response
test_predictor <- test_set[,-1]        # test set predictors

all_response  <- creditset[, 1]        # full set response
all_predictor <- creditset[,-1]        # full set predictors  

remove(Y,msk)

  
  set.seed(825)
  
  creditset$CA_01    <- as.factor(creditset$CA_01)
  test_predictor$CA_01    <- as.factor(test_predictor$CA_01)
  train_set$CA_01 <- as.factor(train_set$CA_01)
  
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
  
  C_RANGE <- 1:10
  sigma <- 1:5
  svmLinGrid <- expand.grid(C = C_RANGE, sigma = sigma)
  
  svmLinControl <- trainControl(method = "cv",
                                search = "random",
                                number = 5,
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
  
  
  svmLin_model <- train(GOOD ~ .,
                        data = train_set,
                        method = "svmRadial",
                        tuneGrid = svmLinGrid,
                        trControl = svmLinControl,
                        metric = "Accuracy")
  
  predicted_svmLin <- predict(svmLin_model,newdata = test_predictor)
  cm_svmLin <- confusionMatrix(predicted,test_response)
  

##########
#Confusion Matrix and Statistics

#Reference
#Prediction  BAD GOOD
#BAD   485  105
#GOOD  127 1027

#Accuracy : 0.867           
#95% CI : (0.8501, 0.8826)
#No Information Rate : 0.6491          
#P-Value [Acc > NIR] : <2e-16          

#Kappa : 0.7056          
#Mcnemar's Test P-Value : 0.168           

#Sensitivity : 0.7925          
#Specificity : 0.9072          
#Pos Pred Value : 0.8220          
#Neg Pred Value : 0.8899          
#Prevalence : 0.3509          
#Detection Rate : 0.2781          
#Detection Prevalence : 0.3383          
#Balanced Accuracy : 0.8499          

#'Positive' Class : BAD             
###########

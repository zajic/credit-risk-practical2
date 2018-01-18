

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

set.seed(34576345)

##############################################################################################
#
# Load data and some more pre-processing of data 
#
##############################################################################################


creditset <- read.csv("csv_files/dataset_manually_imputed_known.csv")

creditset$X <- NULL

creditset$GOOD  <- as.factor(creditset$GOOD)
creditset$CA_01 <- as.factor(creditset$CA_01)

########################
# Split train test set
########################

# Using sample.split from caTools
Y = creditset$GOOD
msk = sample.split(Y,SplitRatio=0.90)
train_set = creditset[msk,]
test_set  = creditset[!msk,]
test_response  <- test_set[, 1]        # test set response
test_predictor <- test_set[,-1]        # test set predictors

all_response  <- creditset[, 1]        # full set response
all_predictor <- creditset[,-1]        # full set predictors  

train_response <- train_set[,1]
train_predictor <- train_set[,-1]

remove(Y)

##############################################################################################
#
# Random Forests
#
##############################################################################################

  
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
  
   ##################
  
  
    NTREE = 1500           # For most cases/tests, 500 to 1000 really is enough.
    MTRY_RANGE = 1:30
    # NOTE: I suspect the randomForest package does automatic categorical var encoding,
    #       after which there may be more than the original 22 variables, so perhaps
    #       mtry could have a wider range than 1:22

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

    rf_model_test <- train(GOOD ~ .,
                           data = train_set,
                           method = "rf",
                           tuneGrid = rfGrid,
                           trControl = rfControl,
                           metric = "ROC",
                           ntree = NTREE)
    
    remove(MTRY_RANGE, NTREE)

    # rf_model <- randomForest(train$GOOD ~ ., data=train,importance=TRUE,ntree=1000)
    
    ############
    # Evaluate
    ############

    # the randomForest's confusion matrix estimate based on OOB error estimate 
    rf_model_test$finalModel$confusion

    # does it agree with test set confusion matrix?
    predicted <- predict(rf_model_test,newdata = test_predictor)
    cm <- confusionMatrix(predicted,test_response,positive="GOOD")

    ################
    # Evaluate ROC
    ################
    
    #ROC
    #The ?best? threshold is considered to be that which maximises the true positives
    #and minimises the false positives. This can be found by the location on
    #the curve closet to the top left corner of the ROC plot.
    library(pROC)
    test_predictor$prob <- predict(rf_model_test,newdata=test_predictor,type="prob")$GOOD
    g <- roc(test_response ~ test_predictor$prob)
    #coords coordinates of ROC curve
    df<-data.frame(t(coords(g, seq(0, 1, 0.01))))
    best<-coords(g, "best")
    require(ggplot2)
    p <- ggplot(df)
    p <- p + geom_line(aes(1-specificity, sensitivity, colour=threshold), size=3) + theme_bw()
    p + geom_abline(intercept=0, slope=1) +
      geom_hline(yintercept=as.numeric(best[3]), colour="darkgrey", linetype="longdash") +
      geom_vline(xintercept = as.numeric(1-best[2]), colour="darkgrey", linetype="longdash") +
      scale_colour_gradient(high="red", low="white") +
      geom_line(aes(1-specificity, sensitivity), colour="blue", alpha=1/3) +
      xlab("1-Specificity (False Positive Rate)") + ylab("Sensitivity (True Positive Rate)") +
      labs(colour="Threshold")
    
    
    test_predictor$probBinary <- rep(NA,length(test_predictor$S_02))
    test_predictor$probBinary[which(test_predictor$prob <= best[1] )] <- "BAD"
    test_predictor$probBinary[which(test_predictor$prob >  best[1])] <- "GOOD"
    
    # best[1]
    # best[1]
    
    cm2 <- confusionMatrix(test_predictor$probBinary,test_response,positive="GOOD")
    
    
    remove(rfGrid, rfControl)

    # NOTE: CONCLUSION: OOB confusion matrix estimate seems to agree with confusion matrix
    #                   using training/test set.
    #                   17 or 19 seems to be the best mtry
    #                   Accuracy around 87%
    #                   use err.rate to gauge tree size
    #                   using importance(), it seems that the post-encoding variable size is 49


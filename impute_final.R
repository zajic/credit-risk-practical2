##############################################################################################
#
# Init 
#
##############################################################################################
library(missForest)
library(mice)
library(caret)
library(caTools)
library(foreach)
library(doParallel)

registerDoParallel(cores=4)

set.seed(3456)

##############################################################################################
#
# Load the dataset and mark obviously NA entries as NA 
#
##############################################################################################

loadDatasetAndPreprocess <- function(filepath) {
  # "." are converted to NA
  creditset <- read.table(filepath, na.strings = ".", sep = ",", header = T)
  
  ####################################################
  # Investigate entries without GOOD or BAD outcomes 
  ####################################################
  
  # NOTE: In report we'll mention about reject inference, and how we need not consider it
  #       if the practical's aim to get high classification scores
  
  # Note the indices of entries with known outcome and those without
  INDICES_KNOWN_OUTCOME   <- which(creditset$app_id >= 0)
  INDICES_UNKNOWN_OUTCOME <- which(creditset$app_id <  0)
  
  # Convert 99999s to NAs
  creditset[which(creditset$time_emp == 99999),8] <- NA
  
  # res_indicator: replace "" with NA
  indexOfEmpty <- which(creditset$res_indicator == "")
  creditset$res_indicator[indexOfEmpty] <- NA
  
  # According to Tom Kelsey's email of 14 March, RBS indicated that negative values for S_02
  # is akin to NA.
  creditset$S_02[which(creditset$S_02 < 0)] <- NA
  
  ##############################################################################################
  #
  # Strip the "day" information from app_date
  #
  # NOTE: Leaving the "year" information in there, on the assumption that we mainly intend
  #       to use this column to check for correlation with time periods, instead of 
  #       correlation with month of the year (ie. JUL2005 is different from JUL2006) 
  #
  ##############################################################################################
  
  creditset$app_date <- substr(creditset$app_date,3,9)
  # Use Date applied column and only keep the month
  # creditset$app_date <- (substr(creditset$app_date, 3, 5))
  
  ##############################################################################################
  #
  # Remove unnecessary columns
  #
  ##############################################################################################
  
  # remove obsolete column
  creditset$BAD <- NULL
  
  ##############################################################################################
  #
  # Mark categorical variables
  #
  ##############################################################################################
  
  # clearly mark categorical variables as such
  # this is especially relevant for CA_01
  if (!is.null(creditset$GOOD)) {
    creditset$GOOD     <- as.factor(creditset$GOOD)
  }
  creditset$CA_01    <- as.factor(creditset$CA_01)
  creditset$app_date <- as.factor(creditset$app_date)
  
  # store the indices of all categorical variables
  INDICES_CATEGORICAL_VARIABLE <- sapply(creditset, is.factor)
  
  results <- list("dataset" = creditset,
                  "indicesKnownOutcome" = INDICES_KNOWN_OUTCOME,
                  "indicesUnknownOutcome" = INDICES_UNKNOWN_OUTCOME,
                  "indicesCategoricalVariables" = INDICES_CATEGORICAL_VARIABLE) 
  
  return(results)
}


####
# Our imputation treatment
####

myImpute <- function(creditset) {
  creditset$I_01 <- NULL
  creditset$I_02 <- NULL
  creditset$D_01 <- NULL
  creditset$D_02 <- NULL

  maxiter <- 30
  ntree <- 100
  
  set.seed(23536)
  
  imputed_final <- missForest(creditset,
                              verbose=TRUE,
                              maxiter=maxiter,
                              ntree=ntree,
                              parallelize = "forests")

  return(imputed_final)
}


loadCreditset <- loadDatasetAndPreprocess("dataset_modelling.csv")
creditset <- loadCreditset$dataset[loadCreditset$indicesKnownOutcome,]

loadHoldout <- loadDatasetAndPreprocess("ALG_HOLDOUT_WITHOUT_OUTCOME.csv")
holdoutset_all <- loadHoldout$dataset
holdoutset <- loadHoldout$dataset[loadHoldout$indicesKnownOutcome,]


########################
# Split train test set
########################

# Using sample.split from caTools
Y = creditset$GOOD
msk = sample.split(Y,SplitRatio=0.75)
train_set = creditset[msk,]
test_set  = creditset[!msk,]
test_response  <- test_set[, which(colnames(creditset)=="GOOD")]          # test set response
test_appid     <- test_set[, which(colnames(creditset)=="app_id")]        # test set app_id
test_predictor <- test_set[,-which(colnames(creditset)=="GOOD")]        
test_predictor <- test_predictor[,-which(colnames(creditset)=="app_id")]  # test set predictors

train_response  <- train_set[, which(colnames(creditset)=="GOOD")]
train_appid     <- train_set[, which(colnames(creditset)=="app_id")]
train_predictor <- train_set[,-which(colnames(creditset)=="GOOD")]
train_predictor <- train_predictor[,-which(colnames(creditset)=="app_id")]

holdout_appid     <- holdoutset[, which(colnames(creditset)=="app_id")]
holdout_predictor <- holdoutset[,-which(colnames(creditset)=="app_id")]

remove(Y)

#####
# Run impute
#####

train_imputed <- myImpute(train_predictor)
test_imputed  <- myImpute(test_predictor)
holdout_imputed <- myImpute(holdout_predictor)

moveLastColumnToFirst <- function(dataset) {
  indices <- c(ncol(dataset), 1:(ncol(dataset) - 1))
  return(dataset[,indices])
}

# add outcome and app_id back to train set
train_imputed_result <- train_imputed$ximp
train_imputed_result$GOOD <- train_response
train_imputed_result <- moveLastColumnToFirst(train_imputed_result)
train_imputed_result$app_id <- train_appid
train_imputed_result <- moveLastColumnToFirst(train_imputed_result)

test_imputed_result <- test_imputed$ximp
test_imputed_result$GOOD <- test_response
test_imputed_result <- moveLastColumnToFirst(test_imputed_result)
test_imputed_result$app_id <- test_appid
test_imputed_result <- moveLastColumnToFirst(test_imputed_result)

holdout_imputed_result <- holdout_imputed$ximp
holdout_imputed_result$app_id <- holdout_appid
holdout_imputed_result <- moveLastColumnToFirst(holdout_imputed_result)

holdout_imputed_with_negAppID <- holdoutset_all
holdout_imputed_with_negAppID$I_01 <- NULL
holdout_imputed_with_negAppID$I_02 <- NULL
holdout_imputed_with_negAppID$D_01 <- NULL
holdout_imputed_with_negAppID$D_02 <- NULL

# add unknown outcome entries back to the holdout set
for (i in 1:nrow(holdout_imputed_result)) {
  appid <- holdout_imputed_result[i,]$app_id
  index <- match(appid, holdout_imputed_with_negAppID$app_id)
  holdout_imputed_with_negAppID[index,] <- holdout_imputed_result[i,]
}

write.csv(train_imputed_result,  file = "dataset_imputed_final_train.csv", row.names=FALSE)
write.csv(test_imputed_result,  file = "dataset_imputed_final_test.csv", row.names=FALSE)
write.csv(holdout_imputed_with_negAppID,  file = "dataset_imputed_final_holdout.csv", row.names=FALSE)

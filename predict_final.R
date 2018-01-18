DATA_FILEPATH_MIDPOINT = "data_holdout_imputed/dataset_manually_imputed_holdout.csv"
DATA_FILEPATH_LATEST = "dataset_imputed_final_holdout.csv"
IMAGE_LATEST_MODEL = "final.RData"
IMAGE_MIDPOINT_MODEL = "90RF.RData"

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

##############################################################################################
#
# Functions
#
##############################################################################################

levelsFromNumToText <- function(dataset) {
  if (!is.null(dataset$GOOD)) {
    dataset$GOOD <- mapvalues(dataset$GOOD,
                              from = c("1", "0"),
                              to = c("GOOD", "BAD"))
  }
  
  dataset$CA_01 <- mapvalues(dataset$CA_01,
                             from = c("1", "2", "3", "4", "5"),
                             to = c("one", "two", "three", "four", "five"))
  
  return(dataset)
}

moveLastColumnToFirst <- function(dataset) {
  indices <- c(ncol(dataset), 1:(ncol(dataset) - 1))
  return(dataset[,indices])
}

##############################################################################################
#
# Load workspace
#
##############################################################################################

load(IMAGE_MIDPOINT_MODEL)

midpointModel <- rf_model_test

load(IMAGE_LATEST_MODEL)

latestModel <- result$caretModel


##############################################################################################
#
# Load data
#
##############################################################################################

creditsetMidpoint <- read.csv(DATA_FILEPATH_MIDPOINT)

creditsetMidpoint$X     <- NULL
creditsetMidpoint$GOOD  <- NULL

creditsetMidpoint$CA_01 <- as.factor(creditsetMidpoint$CA_01)

creditsetMidpoint <- levelsFromNumToText(creditsetMidpoint)



creditsetLatest <- read.csv(DATA_FILEPATH_LATEST)

INDICES_KNOWN_OUTCOME   <- which(creditsetLatest$app_id >= 0)
INDICES_UNKNOWN_OUTCOME <- which(creditsetLatest$app_id <  0)

appid_archive_all <- creditsetLatest$app_id
appid_archive_known <- creditsetLatest[INDICES_KNOWN_OUTCOME,]$app_id

creditsetLatest <- levelsFromNumToText(creditsetLatest)

creditsetLatest$app_id <- NULL

creditsetLatest_All <- creditsetLatest
creditsetLatest <- creditsetLatest[INDICES_KNOWN_OUTCOME,]



##############################################################################################
#
# Predict 2 level and save results
#
##############################################################################################

predictedMidpoint <- predict(midpointModel, newdata = creditsetMidpoint)

stopifnot(nrow(creditsetMidpoint) == length(predictedMidpoint))

creditsetMidpoint$GOOD <- predictedMidpoint

creditsetMidpoint <- moveLastColumnToFirst(creditsetMidpoint)

creditsetMidpoint$app_id <- appid_archive_all

creditsetMidpoint <- moveLastColumnToFirst(creditsetMidpoint)

write.csv(creditsetMidpoint,  file = "predictions_final_midpoint.csv", row.names=FALSE)




predictedLatest <- predict(latestModel, newdata = creditsetLatest)

creditsetLatest$GOOD <- predictedLatest

creditsetLatest <- moveLastColumnToFirst(creditsetLatest)

creditsetLatest$app_id <- appid_archive_known

creditsetLatest <- moveLastColumnToFirst(creditsetLatest)

creditsetLatest_All$GOOD <- rep("",length(creditsetLatest_All$disp_income))

creditsetLatest_All <- moveLastColumnToFirst(creditsetLatest_All)

creditsetLatest_All$app_id <- appid_archive_all

creditsetLatest_All <- moveLastColumnToFirst(creditsetLatest_All)


creditsetLatest$GOOD <- as.character(creditsetLatest$GOOD)


# add unknown outcome entries back to the holdout set
for (i in 1:nrow(creditsetLatest)) {
  appid <- creditsetLatest[i,]$app_id
  index <- match(appid, creditsetLatest_All$app_id)
  creditsetLatest_All[index,] <- creditsetLatest[i,]
}

creditsetLatest$GOOD <- as.factor(creditsetLatest$GOOD)

write.csv(creditsetLatest_All, file = "predictions_final_latest.csv", row.names=FALSE)




table(creditsetMidpoint[INDICES_KNOWN_OUTCOME,]$GOOD)

table(creditsetLatest$GOOD)

##############################################################################################
#
# Load data
#
##############################################################################################

creditsetMidpoint <- read.csv(DATA_FILEPATH_MIDPOINT)

creditsetMidpoint$X     <- NULL
creditsetMidpoint$GOOD  <- NULL

creditsetMidpoint$CA_01 <- as.factor(creditsetMidpoint$CA_01)

creditsetMidpoint <- levelsFromNumToText(creditsetMidpoint)



creditsetLatest <- read.csv(DATA_FILEPATH_LATEST)

INDICES_KNOWN_OUTCOME   <- which(creditsetLatest$app_id >= 0)
INDICES_UNKNOWN_OUTCOME <- which(creditsetLatest$app_id <  0)

appid_archive_all <- creditsetLatest$app_id
appid_archive_known <- creditsetLatest[INDICES_KNOWN_OUTCOME,]$app_id

creditsetLatest <- levelsFromNumToText(creditsetLatest)

creditsetLatest$app_id <- NULL

creditsetLatest_All <- creditsetLatest
creditsetLatest <- creditsetLatest[INDICES_KNOWN_OUTCOME,]

##############################################################################################
#
# Predict 3 level and save results
#
##############################################################################################

midpointThresh <- 0.66225
midpointRange <- 0.15

latestThresh <- 0.601
latestRange <- 0.15

predictedMidpoint <- predict(midpointModel, newdata = creditsetMidpoint, type = "prob")
predictedMidpoint$BAD <- NULL

temp <- rep(NA,length(predictedMidpoint))
temp[which(predictedMidpoint <= midpointThresh)] <- "BAD"
temp[which(predictedMidpoint >  midpointThresh + midpointRange)] <- "GOOD"
temp[which((predictedMidpoint >  midpointThresh) & (predictedMidpoint <=  midpointThresh + midpointRange))] <- "PASS"

creditsetMidpoint$GOOD <- temp

remove(temp)

creditsetMidpoint <- moveLastColumnToFirst(creditsetMidpoint)

creditsetMidpoint$app_id <- appid_archive_all

creditsetMidpoint <- moveLastColumnToFirst(creditsetMidpoint)

write.csv(creditsetMidpoint,  file = "predictions_final_midpoint_pass.csv", row.names=FALSE)





predictedLatest <- predict(latestModel, newdata = creditsetLatest, type = "prob")
predictedLatest$BAD <- NULL

temp <- rep(NA,length(predictedLatest))
temp[which(predictedLatest <= latestThresh)] <- "BAD"
temp[which(predictedLatest >  latestThresh + latestRange)] <- "GOOD"
temp[which((predictedLatest >  latestThresh) & (predictedLatest <=  latestThresh + latestRange))] <- "PASS"

creditsetLatest$GOOD <- temp

remove(temp)

creditsetLatest <- moveLastColumnToFirst(creditsetLatest)

creditsetLatest$app_id <- appid_archive_known

creditsetLatest <- moveLastColumnToFirst(creditsetLatest)

creditsetLatest_All$GOOD <- rep("",length(creditsetLatest_All$disp_income))

creditsetLatest_All <- moveLastColumnToFirst(creditsetLatest_All)

creditsetLatest_All$app_id <- appid_archive_all

creditsetLatest_All <- moveLastColumnToFirst(creditsetLatest_All)


creditsetLatest$GOOD <- as.character(creditsetLatest$GOOD)


# add unknown outcome entries back to the holdout set
for (i in 1:nrow(creditsetLatest)) {
  appid <- creditsetLatest[i,]$app_id
  index <- match(appid, creditsetLatest_All$app_id)
  creditsetLatest_All[index,] <- creditsetLatest[i,]
}

creditsetLatest$GOOD <- as.factor(creditsetLatest$GOOD)

write.csv(creditsetLatest_All, file = "predictions_final_latest_pass.csv", row.names=FALSE)

table(creditsetMidpoint[INDICES_KNOWN_OUTCOME,]$GOOD)

table(creditsetLatest$GOOD)
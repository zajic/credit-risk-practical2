DATA_FILEPATH = "data_holdout_imputed/dataset_manually_imputed_holdout.csv"
WORKSPACE_IMAGE_PATH = "90RF.RData"

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
# Load workspace
#
##############################################################################################

load(WORKSPACE_IMAGE_PATH)

##############################################################################################
#
# Load functions
#
##############################################################################################

# Create a confusion matrix from the given outcomes, whose rows correspond
# to the actual and the columns to the predicated classes.
createConfusionMatrix <- function(act, pred) {
  tpGood = 0;
  tpBad = 0;
  good_pass=0;
  good_bad = 0;
  bad_good = 0;
  bad_pass = 0;
  
  for (i in 1:length(act)){
    
    if (pred[i] == "GOOD" & act[i] == "GOOD"){
      tpGood = tpGood+1;
    }
    if (act[i] == pred[i] & act[i] == "BAD"){
      tpBad = tpBad+1;
    }
    if (act[i] == "GOOD" & pred[i] == "PASS"){
      good_pass = good_pass+1;
    }
    if (act[i] == "GOOD" & pred[i] == "BAD"){
      good_bad = good_bad + 1;
    }
    
    if (act[i] == "BAD" & pred[i] == "GOOD"){
      bad_good = bad_good + 1;
    }
    if (act[i] == "BAD" & pred[i] == "PASS"){
      bad_pass = bad_pass + 1;
    }
    
  }
  
  acc = (tpGood + tpBad) / length(act)
  spec = tpBad / (tpBad + bad_good + bad_pass)
  sens = tpGood / (tpGood + good_bad + good_pass)
  markedAsPass = (bad_pass + good_pass)/length(act)
  print ("     GOOD    PASS    BAD")
  print(paste("GOOD ",tpGood,"  ",good_pass,"   ",good_bad))
  print("PASS 0      0       0")
  print(paste("BAD ",bad_good,"   ",bad_pass,"   ",tpBad))
  
  print(paste("Accuracy: ",acc))
  print(paste("False positive",bad_good/(bad_good + bad_pass + tpBad)))
  print(paste("Specificity: ",spec))
  print(paste("Sensitivity: ",sens))
  print(paste("Pass: ",markedAsPass))
  
}

#try predicting "PASS" on different ranges above and below the threshold
differentRanges <- function(){
  
  test_predictor$probBinary <- rep(NA,length(test_predictor$S_02))
  a <- c(0.2,0.15,0.1,0.05)
  b <- a
  c <- expand.grid(a,b)
  
  for (i in 1:16){
    print(c[i,1])
    print(c[i,2])
    
    test_predictor$probBinary[which(test_predictor$prob <= best[1] - c[i,1] )] <- "BAD"
    test_predictor$probBinary[which(test_predictor$prob >  best[1] + c[i,2])] <- "GOOD"
    test_predictor$probBinary[which((test_predictor$prob >  best[1] - c[i,1]) & (test_predictor$prob <  best[1] + c[i,2]))] <- "PASS"
    
    createConfusionMatrix(test_response,test_predictor$probBinary)
    
  }
  
}

#try predicting "PASS" on different ranges above the threshold
diffRangesOneSided <- function(){
  
  test_predictor$probBinary <- rep(NA,length(test_predictor$S_02))
  c <- c(0.25,0.2,0.15,0.1,0.05)
  
  
  for (i in 1:5) {
    print(c[i])
    
    test_predictor$probBinary[which(test_predictor$prob <= best[1])] <- "BAD"
    test_predictor$probBinary[which(test_predictor$prob >  best[1] + c[i])] <- "GOOD"
    test_predictor$probBinary[which((test_predictor$prob >  best[1]) & (test_predictor$prob <  best[1] + c[i]))] <- "PASS"
    
    createConfusionMatrix(test_response,test_predictor$probBinary)
    
  }
  
}

##############################################################################################
#
# Load data and some more pre-processing of data 
#
##############################################################################################


creditset <- read.csv(DATA_FILEPATH)

creditset$X     <- NULL
creditset$GOOD  <- NULL

creditset$CA_01 <- as.factor(creditset$CA_01)

##############################################################################################
#
# Some more pre-processing 
#
##############################################################################################
  
library(plyr)
# Full
creditset$CA_01 <- mapvalues(creditset$CA_01,
                             from = c("1", "2", "3", "4", "5"),
                             to = c("one", "two", "three", "four", "five"))
  
revalue(creditset$CA_01, c("1"="one", "2"="two", "3"="three", "4"="four", "5"="five"))

##############################################################################################
#
# Prediction and save results
#
##############################################################################################

predicted <- predict(rf_model_test,newdata = test_predictor,type="prob")
predicted$BAD <- NULL


##############################################################################################
#
# Evaluate ROC to calculate threshold
#
##############################################################################################

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

##############################################################################################
#
# Try different ranges for "PASS" values (above, above and below the threshold)
#
##############################################################################################

differentRanges()
diffRangesOneSided()

##############################################################################################
#
# Predict values
#
##############################################################################################

passLimit <- 0.15

print(best[1])

test_predictor$probBinary[which(test_predictor$prob <= best[1])] <- "BAD"
test_predictor$probBinary[which(test_predictor$prob >  best[1] + passLimit)] <- "GOOD"
test_predictor$probBinary[which((test_predictor$prob >  best[1]) & (test_predictor$prob <  best[1] + passLimit))] <- "PASS"

createConfusionMatrix(test_response,test_predictor$probBinary)








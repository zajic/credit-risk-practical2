##############################################################################################
#
# Run? 
#
##############################################################################################

RUN_FINAL_IMPUTATION = FALSE
RUN_MANUAL_IMPUTATION = TRUE
RUN_MANUAL_IMPUTATION_TESTS = TRUE
USE_HOLDOUT_DATA = TRUE

##############################################################################################
#
# Init 
#
##############################################################################################

library(mice)
library(missForest)

library(foreach)
library(doParallel)

registerDoParallel(cores=4)

set.seed(3456)

##############################################################################################
#
# Load the dataset and mark obviously NA entries as NA 
#
##############################################################################################

# "." are converted to NA
creditset <- read.table("csv_files/dataset_modelling.csv", na.strings = ".", sep = ",", header = T)

####################################################
# Investigate entries without GOOD or BAD outcomes 
####################################################

# NOTE: In report we'll mention about reject inference, and how we need not consider it
#       if the practical's aim to get high classification scores

# Verify that those with negative app_id are those with no GOOD or BAD outcome
a <- which(creditset$app_id < 0)
b <- which(is.na(creditset$GOOD) | is.na(creditset$BAD))
stopifnot(identical(a,b))
remove(a,b)

# Note the indices of entries with known outcome and those without
INDICES_KNOWN_OUTCOME   <- which(creditset$app_id >= 0)
INDICES_UNKNOWN_OUTCOME <- which(creditset$app_id <  0)

# Split the data into a set with finished applications and a set with unfinished ones
# splitted <- split(creditset, creditset$app_id < 0)

# Save unknown outcome dataset to file, in case we need it later
# write.csv(splitted$"TRUE",  file = "dataset_UnknownOutcome.csv")

# Use the known outcome dataset as main dataset
# creditset <- splitted$"FALSE"
# remove(splitted)

if (USE_HOLDOUT_DATA == TRUE) {
temp <- nrow(creditset)

creditset2 <- read.table("csv_files/ALG_HOLDOUT_WITHOUT_OUTCOME.csv", na.strings = ".", sep = ",", header = T)
creditset2$GOOD <- rep(NA,length(creditset2$S_02))
creditset2$BAD  <- rep(NA,length(creditset2$S_02))

creditset <- rbind(creditset, creditset2)

INDICES_HOLDOUT_DATA <- c((temp + 1):nrow(creditset))
remove(temp)
remove(creditset2)
}

print(noquote(""))
print(noquote("After converting . to NA, the # of complete cases:"))
print(length(complete.cases(creditset)[complete.cases(creditset)==TRUE]))

# Convert 99999s to NAs
creditset[which(creditset$time_emp == 99999),8] <- NA
print(noquote(""))
print(noquote("After converting 99999 in time_emp to NA, the # of complete cases:"))
print(length(complete.cases(creditset)[complete.cases(creditset)==TRUE]))

# res_indicator: replace "" with NA
indexOfEmpty <- which(creditset$res_indicator == "")
creditset$res_indicator[indexOfEmpty] <- NA
print(noquote(""))
print(noquote("After converting blank in res_indicator to NA, the # of complete cases:"))
print(length(complete.cases(creditset)[complete.cases(creditset)==TRUE]))

# According to Tom Kelsey's email of 14 March, RBS indicated that negative values for S_02
# is akin to NA.
creditset$S_02[which(creditset$S_02 < 0)] <- NA
print(noquote(""))
print(noquote("After converting -1 in S_02 to NA, the # of complete cases:"))
print(length(complete.cases(creditset)[complete.cases(creditset)==TRUE]))

remove(indexOfEmpty)

# Quick check of the count and percentage of NAs in each column
print(noquote(""))
print(noquote("Count of NAs in each column:"))
print(colSums(is.na(creditset)))
#       app_id          GOOD           BAD      app_date   disp_income 
#            0          4149          4149             0             0
#
#     occ_code      cust_age      time_emp res_indicator          I_01 
#            0             0          1847           750          8044
#
#         I_02          I_03          I_04          D_01         ER_01 
#         7555          3772          4590          7157          4198 
#
#        ER_02          I_05          D_02          I_06          P_01 
#         4327          3588          8622             0           802 
#
#         S_01         CA_03         CA_02         CA_01          S_02 
#         5106             0          4886             0          5419 
print(noquote(""))
print(noquote("Ratio of NAs in each column:"))
print(colSums(is.na(creditset)) / nrow(creditset))
#       app_id          GOOD           BAD      app_date   disp_income 
#   0.00000000    0.41648263    0.41648263    0.00000000    0.00000000 
#
#     occ_code      cust_age      time_emp res_indicator          I_01 
#   0.00000000    0.00000000    0.18540454    0.07528609    0.80746838 
#
#         I_02          I_03          I_04          D_01         ER_01 
#   0.75838185    0.37863883    0.46075085    0.71843003    0.42140133 
#
#        ER_02          I_05          D_02          I_06          P_01 
#   0.43435053    0.36016864    0.86548886    0.00000000    0.08050592 
#
#         S_01         CA_03         CA_02         CA_01          S_02 
#   0.51254768    0.00000000    0.49046376    0.00000000    0.54396707

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
creditset$app_id <- NULL

##############################################################################################
#
# Mark categorical variables
#
##############################################################################################

# clearly mark categorical variables as such
# this is especially relevant for CA_01
creditset$GOOD     <- as.factor(creditset$GOOD)
creditset$CA_01    <- as.factor(creditset$CA_01)
creditset$app_date <- as.factor(creditset$app_date)

# store the indices of all categorical variables
INDICES_CATEGORICAL_VARIABLE <- sapply(creditset, is.factor)

##############################################################################################
#
# Output processed dataset
#
##############################################################################################

write.csv(creditset, file = "dataset_transformed.csv")

##############################################################################################
#
# Test run of different imputation methods
#
##############################################################################################

#######################################################################
# Generate a test dataset from complete cases using mice's ampute()
# Final product: amputed_mice
#######################################################################

# NOTE: mice does multiple imputed data sets, and I think the idea is to pool them
#       when fitting. I'm not sure our method here of comparing it to missForest's resultant
#       imputated set is entirely fair.

# mice requires that categorical variables are converted to numeric format
to_numeric <- creditset
# For $GOOD and $CA_01, let's try to preserve the character format
to_numeric["GOOD"] <- lapply(to_numeric["GOOD"], function(x) as.numeric(as.character(x)))
to_numeric["CA_01"] <- lapply(to_numeric["CA_01"], function(x) as.numeric(as.character(x)))
# For the others, let's use R's internal integer IDs
# NOTE: must find a way to note down the translation
indices <- sapply(to_numeric, is.factor)
to_numeric[indices] <- lapply(to_numeric[indices], function(x) as.numeric(x))

# Convert Month MMM to numeric
# creditset$app_date <- match(creditset$app_date,toupper(month.abb))

# Calculate different categories for occ_code
# uq_occ <- unique(creditset$occ_code)

# Assign each of them a different number
# creditset$occ_code<- match(creditset$occ_code,(uq_occ))

# Calculate different categories for res_indicator
# uq_res <- unique(creditset$res_indicator)

# Assign each of them a different number
# creditset$res_indicator<- match(creditset$res_indicator,(uq_res))

# Find the complete.cases
complete_cases   <- to_numeric[complete.cases(to_numeric),]
incomplete_cases <- to_numeric[!complete.cases(to_numeric),]

# Find the patterns of NA in the dataset
patterns <- !is.na(incomplete_cases)

# Check the number of occurences of each unique pattern
library(plyr)
pattern_count <- count(patterns)
sum(pattern_count$freq > 5)   # 276
sum(pattern_count$freq > 10)  # 159
sum(pattern_count$freq > 50)  # 34
sum(pattern_count$freq > 100) # 15
sum(pattern_count$freq > 150) # 9
detach("package:plyr", unload=TRUE)

# We don't have many complete cases to work with, so let's just only use those 9 patterns
# with more than 150 occurrences
patterns <- pattern_count[which(pattern_count$freq > 150),]

# Calculate the relative probability of these patterns
n <- sum(patterns$freq)
patterns$probability <- patterns$freq / n
patterns$probability <- round(patterns$probability,2)

# Due to floating point rounding error the probilities won't add up to 1.
# I'll just arbitratily tack that on to the first pattern.
sum <- sum(patterns$probability)
remainder <- (1 - sum)
patterns$probability[1] <- patterns$probability[1] + remainder

# 0 is with missing values (!is.na == FALSE), 1 is without missing values (!is.na == TRUE)
patterns[patterns==TRUE]  <- 1
patterns[patterns==FALSE] <- 0

# Ready the data to be fed to mice's ampute()
freq <- patterns$probability
patterns <- patterns[,1:23]

mice_mads <- ampute(complete_cases,
                    prop = round((nrow(incomplete_cases)/nrow(creditset)),2),
                    patterns = patterns,
                    freq = freq)

amputed_mice <- mice_mads$amp

# transform categorical variables from numeric back to categorical variables
amputed_mice[INDICES_CATEGORICAL_VARIABLE] <- lapply(amputed_mice[INDICES_CATEGORICAL_VARIABLE],
                                                     function(x) as.factor(x))


remove(patterns, pattern_count, sum, remainder, freq, to_numeric, indices)


##########################################################################
# Generate a test dataset from complete cases using missForests's prodNA 
# Final product: amputed_missForest
##########################################################################

amputed_missForest <- prodNA(complete_cases, noNA=0.2)

# transform categorical variables from numeric back to categorical variables
amputed_missForest[INDICES_CATEGORICAL_VARIABLE] <- lapply(amputed_missForest[INDICES_CATEGORICAL_VARIABLE],
                                                     function(x) as.factor(x))

#############
# With MICE
#############


###################
# With missForest
###################


remove(complete_cases, incomplete_cases)

##############################################################################################
#
# Final, real imputation
#
##############################################################################################

if (RUN_FINAL_IMPUTATION == TRUE) {

print(noquote("Running final imputation. Will overwrite files."))

maxiter <- 10
ntree <- 150


set.seed(23536)

imputed_final <- missForest(creditset,
                            verbose=TRUE,
                            maxiter=maxiter,
                            ntree=ntree,
                            parallelize = "forests")

write.csv(imputed_final$ximp, file = "dataset_imputed.csv")

write.csv(imputed_final$ximp[INDICES_KNOWN_OUTCOME,],  file = "dataset_imputed_known.csv")
write.csv(imputed_final$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "dataset_imputed_unknown.csv")
write.csv(imputed_final$ximp[INDICES_HOLDOUT_DATA,],  file = "dataset_imputed_holdout.csv")

}

##############################################################################################
#
# Experiment: Manual / artificially done imputation
#
##############################################################################################

if (RUN_MANUAL_IMPUTATION == TRUE) {
  
  # The experimentation is run with original dataset + holdout dataset = 14943 rows
  
  # The experimentation makes many assumptions that are not supported by any evidence, except
  # our common sense intuition regarding the data collection process.
  
  # The resultant dataset is meant to be experimental, to be compared with an automatically
  # imputed dataset via fitting models with each and comparing predictive ability of said
  # resultant models.
  
  ########
  # I_01
  ########
  
  # Trend: I_01 likely to be NA if I_03 > 6
  #
  # Intuition: Contrary to data dictionary, I_01 should be interpreted
  #            as "no. of accounts opened in the last SIX months".
  #            Those with I_03 > 6 likely should have 0 as I_01.
  #
  # With our manual imputation, let's operate on this assumption.
  
  a <- which(creditset$I_03 > 6 & (!is.na(creditset$I_01)))
  length(a) # 0         (with holdout data)
  b <- which(creditset$I_03 > 3 & (!is.na(creditset$I_01)))
  length(b) # 1166      (with holdout data)
  stopifnot(length(a)==0)
  remove(a,b)

  # Effecting the change
  treatI01 <- function(creditset) {
    creditset$I_01[is.na(creditset$I_01) & (creditset$I_03 > 6)] <- 0
    return(creditset)
  }
  
  ########
  # S_01
  ########
  
  # S_01 seems like an important predictor
  #
  # If we impute it automatically, the NAs will all mostly likely become ~1.
  #
  # Assume that most of those NAs are supposed to be 0, this means we'll lose some information, namely
  # the different between S_01 = 0 and S_01 = 1.
  #
  # Let's operate on this assumption for now, and make the NAs in S_01 to be 0, except where seemingly
  # contradicted by info from other columns.
  
  # These are the entries whose S_01 seems to disagree with S_02 (according to my understanding of the
  # variable meanings).
  a <- which(creditset$S_02 <= 3 & is.na(creditset$S_01))
  length(a) # 410       (with holdout data)
  # Even if we expand the definition so that S_01 means "# of credit search recorded", the number entries
  # whose S_01 disagrees with S_02 is small.
  b <- which(creditset$S_02 > 0 & is.na(creditset$S_01))
  length(b) # 643       (with holdout data)
  # Therefore, for now there's no definite indication that making most S_01 NAs into 0 is a wrong move.
  remove(a,b)
  
  treatS01 <- function(creditset) {
    INDICES_KEEP_NA <- which(creditset$S_02 > 0 & is.na(creditset$S_01))
    
    # Effecting the change
    creditset$S_01[is.na(creditset$S_01)] <- 0
    creditset$S_01[INDICES_KEEP_NA] <- NA
    
    remove(INDICES_KEEP_NA)
    return(creditset)
  }

  
  ########
  # S_02
  ########

  # Conversely, one can assume that most of the NAs in S_02 mean that no credit search took place within
  # the last 12 months. In that case, we should set the NAs to be a high number equal to or more than 12.
  # However, a normal imputation would place the average of S_02 to ~2.
  
  # Let's turn NAs in S_02 into 12, except where directly contradicted by S_01, in which case we'll leave it
  # as NA.
  
  INDICES_KEEP_NA <- which(is.na(creditset$S_02) & (creditset$S_01 > 0 & !is.na(creditset$S_01)))
  length(INDICES_KEEP_NA) # 1135      (with holdout data)
  
  treatS02 <- function(creditset) {
    INDICES_KEEP_NA <- which(is.na(creditset$S_02) & (creditset$S_01 > 0 & !is.na(creditset$S_01)))
    
    # Effecting the change
    creditset$S_02[is.na(creditset$S_02)] <- 12
    creditset$S_02[INDICES_KEEP_NA] <- NA
    
    remove(INDICES_KEEP_NA) 
    
    return(creditset)
  }

  
  #################
  # I_02 and D_02
  #################
  
  # By looking at the variable meanings, it would seem that I_02 and D_02 have some relations
  # According to common sense, useful derived features would be (D_02) / (I_02) and (D_02 - I_02).
  # Supposedly these features can be learned/deduced by some machine learning algorithms. However, computing them
  # and adding them to the dataset manually may help. This however, comes at a cost of higher input dimensions.
  # There is a tradeoff here. We'll try it with manually computed features.
  
  addNewFeature1 <- function(creditset) {
    creditset$NewFeature1 <- (creditset$D_02) / (creditset$I_02)
    return(creditset)
  }
 
  addNewFeature2 <- function(creditset) {
    creditset$NewFeature2 <- (creditset$I_02) - (creditset$D_02)
    return(creditset)
  }
  
  addNewFeature2_original <- function(creditset) {
    creditset$NewFeature2 <- (creditset$D_02) - (creditset$I_02)
    return(creditset)
  }
  
  # I'd be tempted to extract features from examining D_02's relation with disp_income as well, except that I think
  # disp_income has data quality issues, and using it to manually compute features may actually lead to more noise
  # and confusion for the machine learning algorithm
  
  #########
  # CA_02
  #########
  
  # CA_03 pretty much has no NAs. Let's trust that data, and see if we can use it to manipulate CA_02.
  # CA_02 means "Applicant: total balance of all live current accounts." For those people who have no
  # live current accounts at all (CA_03 = 0), what are their CA_02?
  
  a <- which(creditset$CA_03 == 0 & (creditset$CA_02 > 0 & !is.na(creditset$CA_02)))
  length(a) # 0         (with holdout data)
  
  b <- which(is.na(creditset$CA_02) & creditset$CA_03 == 0)
  length(b) # 7316     (with holdout data)
  
  c <- which(!is.na(creditset$CA_02) & creditset$CA_02 == 0 & creditset$CA_03 == 0)
  length(b) # 312       (with holdout data)

  remove(a,b,c)
  
  # For those without a live current account (CA_03 = 0), none has an account balance greater than 0, as expected.
  # Some of those has an account balanced marked 0. The overwhelming majority has an account balance marked NA.
  
  # We'll leave this unchanged for now.
  
  ############
  # time_emp
  ############
  
  
  # time_emp looks fishy, will very old people having very low time_emp,
  # and a maximum time_emp of 6. Let's treat it as categorical variable.
  
  asFactorTimeEmp <- function(creditset) {
    creditset$time_emp <- as.factor(creditset$time_emp)
    return(creditset)
  }

  
  ###########################
  # Unused and undocumented
  ###########################
  
  a <- which(is.na(creditset$I_04) & (!is.na(creditset$I_01) | !is.na(creditset$I_02) | !is.na(creditset$I_03)))
  
  b <- which( (creditset$I_04 > creditset$I_05) | (!is.na(creditset$I_04) & is.na(creditset$I_05)))
  
  c <- which((creditset$P_01 == 0 | is.na(creditset$P_01)) & (creditset$I_02 == 0 | !is.na(creditset$I_02)))
  
  d <- which(creditset$ER_02 > creditset$ER_01)
  
  e <- which(creditset$S_02 <= 3 & is.na(creditset$S_01))
  
  remove(a,b,c,d,e)
  
  #####################################
  # Some tests
  #####################################
  
  if (RUN_MANUAL_IMPUTATION_TESTS == TRUE) {
    
    maxiter <- 20
    ntree <- 100
    
    manualset <- creditset
    set.seed(23536)
    testBaseline <- missForest(manualset,
                               variablewise = TRUE,
                               replace = TRUE,
                               verbose=TRUE,
                               maxiter=maxiter,
                               ntree=ntree,
                               parallelize = "forests")
    write.csv(testBaseline$ximp[INDICES_KNOWN_OUTCOME,], file = "./data_manimp_tests/manImp_testBaseline_known.csv", row.names=FALSE)
    write.csv(testBaseline$ximp[INDICES_UNKNOWN_OUTCOME,], file = "./data_manimp_tests/manImp_testBaseline_unknown.csv", row.names=FALSE)
    write.csv(testBaseline$ximp[INDICES_HOLDOUT_DATA,], file = "./data_manimp_tests/manImp_testBaseline_holdout.csv", row.names=FALSE)
    
    maxiter <- 20
    ntree <- 100
    
    manualset <- creditset
    manualset <- treatI01(manualset)
    set.seed(23536)
    testI01 <- missForest(manualset,
                          variablewise = TRUE,
                          replace = TRUE,
                          verbose=TRUE,
                          maxiter=maxiter,
                          ntree=ntree,
                          parallelize = "forests")
    write.csv(testI01$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testI01_known.csv", row.names=FALSE)
    write.csv(testI01$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testI01_unknown.csv", row.names=FALSE)
    write.csv(testI01$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testI01_holdout.csv", row.names=FALSE)
    
    maxiter <- 20
    ntree <- 100
    
    manualset <- creditset
    manualset <- treatS01(manualset)
    set.seed(23536)
    testS01 <- missForest(manualset,
                          variablewise = TRUE,
                          replace = TRUE,
                          verbose=TRUE,
                          maxiter=maxiter,
                          ntree=ntree,
                          parallelize = "forests")
    write.csv(testS01$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS01_known.csv", row.names=FALSE)
    write.csv(testS01$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS01_unknown.csv", row.names=FALSE)
    write.csv(testS01$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testS01_holdout.csv", row.names=FALSE)
    
    manualset <- creditset
    manualset <- treatS02(manualset)
    set.seed(23536)
    testS02 <- missForest(manualset,
                          variablewise = TRUE,
                          replace = TRUE,
                          verbose=TRUE,
                          maxiter=maxiter,
                          ntree=ntree,
                          parallelize = "forests")
    write.csv(testS02$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS02_known.csv", row.names=FALSE)
    write.csv(testS02$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS02_unknown.csv", row.names=FALSE)
    write.csv(testS02$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testS02_holdout.csv", row.names=FALSE)

    manualset <- creditset
    manualset <- treatS01(manualset)
    manualset <- treatS02(manualset)
    set.seed(23536)
    testS01S02 <- missForest(manualset,
                             variablewise = TRUE,
                             replace = TRUE,
                             verbose=TRUE,
                             maxiter=maxiter,
                             ntree=ntree,
                             parallelize = "forests")
    write.csv(testS01S02$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS01S02_known.csv", row.names=FALSE)
    write.csv(testS01S02$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testS01S02_unknown.csv", row.names=FALSE)
    write.csv(testS01S02$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testS01S02_holdout.csv", row.names=FALSE)

    manualset <- creditset
    manualset <- addNewFeature1(manualset)
    set.seed(23536)
    testNewFeat1 <- missForest(manualset,
                               variablewise = TRUE,
                               replace = TRUE,
                               verbose=TRUE,
                               maxiter=maxiter,
                               ntree=ntree,
                               parallelize = "forests")
    write.csv(testNewFeat1$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1_known.csv", row.names=FALSE)
    write.csv(testNewFeat1$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1_unknown.csv", row.names=FALSE)
    write.csv(testNewFeat1$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat1_holdout.csv", row.names=FALSE)
    
    testNewFeat1ImputeOnly <- testNewFeat1$ximp
    testNewFeat1ImputeOnly$NewFeature1 <- NULL
    write.csv(testNewFeat1ImputeOnly[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1ImputeOnly_known.csv", row.names=FALSE)
    write.csv(testNewFeat1ImputeOnly[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1ImputeOnly_unknown.csv", row.names=FALSE)
    write.csv(testNewFeat1ImputeOnly[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat1ImputeOnly_holdout.csv", row.names=FALSE)
    
    manualset <- creditset
    manualset <- addNewFeature2(manualset)
    set.seed(23536)
    testNewFeat2 <- missForest(manualset,
                               variablewise = TRUE,
                               replace = TRUE,
                               verbose=TRUE,
                               maxiter=maxiter,
                               ntree=ntree,
                               parallelize = "forests")
    write.csv(testNewFeat2$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2_known.csv", row.names=FALSE)
    write.csv(testNewFeat2$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2_unknown.csv", row.names=FALSE)
    write.csv(testNewFeat2$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat2_holdout.csv", row.names=FALSE)
    
    testNewFeat2ImputeOnly <- testNewFeat2$ximp
    testNewFeat2ImputeOnly$NewFeature2 <- NULL
    write.csv(testNewFeat2ImputeOnly[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2ImputeOnly_known.csv", row.names=FALSE)
    write.csv(testNewFeat2ImputeOnly[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2ImputeOnly_unknown.csv", row.names=FALSE)
    write.csv(testNewFeat2ImputeOnly[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat2ImputeOnly_holdout.csv", row.names=FALSE)
    
    manualset <- creditset
    manualset <- addNewFeature1(manualset)
    manualset <- addNewFeature2(manualset)
    set.seed(23536)
    testNewFeat1NewFeat2 <- missForest(manualset,
                                       variablewise = TRUE,
                                       replace = TRUE,
                                       verbose=TRUE,
                                       maxiter=maxiter,
                                       ntree=ntree,
                                       parallelize = "forests")
    write.csv(testNewFeat1NewFeat2$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2_known.csv", row.names=FALSE)
    write.csv(testNewFeat1NewFeat2$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2_unknown.csv", row.names=FALSE)
    write.csv(testNewFeat1NewFeat2$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2_holdout.csv", row.names=FALSE)

    testNewFeat1NewFeat2ImputeOnly <- testNewFeat1NewFeat2$ximp
    testNewFeat1NewFeat2ImputeOnly$NewFeature1 <- NULL
    testNewFeat1NewFeat2ImputeOnly$NewFeature2 <- NULL
    write.csv(testNewFeat1NewFeat2ImputeOnly[INDICES_KNOWN_OUTCOME,],
              file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2ImputeOnly_known.csv",
              row.names=FALSE)
    write.csv(testNewFeat1NewFeat2ImputeOnly[INDICES_UNKNOWN_OUTCOME,],
              file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2ImputeOnly_unknown.csv",
              row.names=FALSE)
    write.csv(testNewFeat1NewFeat2ImputeOnly[INDICES_HOLDOUT_DATA,],
              file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2ImputeOnly_holdout.csv",
              row.names=FALSE)
    
    manualset <- creditset
    manualset <- asFactorTimeEmp(manualset)
    set.seed(23536)
    testTimeEmp <- missForest(manualset,
                              variablewise = TRUE,
                              replace = TRUE,
                              verbose=TRUE,
                              maxiter=maxiter,
                              ntree=ntree,
                              parallelize = "forests")
    write.csv(testTimeEmp$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testTimeEmp_known.csv", row.names=FALSE)
    write.csv(testTimeEmp$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testTimeEmp_unknown.csv", row.names=FALSE)
    write.csv(testTimeEmp$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testTimeEmp_holdout.csv", row.names=FALSE)
        
    remove(maxiter, ntree)
    remove(manualset)
    
    # New Features: before or after?
    maxiter <- 20
    ntree <- 100
    
    manualset <- creditset

    set.seed(23536)
    vanilla <- missForest(manualset,
                          variablewise = TRUE,
                          replace = TRUE,
                          verbose=TRUE,
                          maxiter=maxiter,
                          ntree=ntree,
                          parallelize = "forests")

    NewFeat1_AppendAfter <- vanilla$ximp
    NewFeat1_AppendAfter <- addNewFeature1(NewFeat1_AppendAfter)
    write.csv(NewFeat1_AppendAfter[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1After_known.csv", row.names=FALSE)
    write.csv(NewFeat1_AppendAfter[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1After_unknown.csv", row.names=FALSE)
    write.csv(NewFeat1_AppendAfter[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat1After_holdout.csv", row.names=FALSE)
    
    NewFeat2_AppendAfter <- vanilla$ximp
    NewFeat2_AppendAfter <- addNewFeature2(NewFeat2_AppendAfter)
    write.csv(NewFeat2_AppendAfter[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2After_known.csv", row.names=FALSE)
    write.csv(NewFeat2_AppendAfter[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat2After_unknown.csv", row.names=FALSE)
    write.csv(NewFeat2_AppendAfter[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat2After_holdout.csv", row.names=FALSE)
    
    NewFeat1NewFeat2_AppendAfter <- vanilla$ximp
    NewFeat1NewFeat2_AppendAfter <- addNewFeature1(NewFeat1NewFeat2_AppendAfter)
    NewFeat1NewFeat2_AppendAfter <- addNewFeature2(NewFeat1NewFeat2_AppendAfter)
    write.csv(NewFeat1NewFeat2_AppendAfter[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2After_known.csv", row.names=FALSE)
    write.csv(NewFeat1NewFeat2_AppendAfter[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2After_unknown.csv", row.names=FALSE)
    write.csv(NewFeat1NewFeat2_AppendAfter[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2After_holdout.csv", row.names=FALSE)
    
    NewFeat1_ReadjustKnown <- read.csv("./data_manimp_tests/manImp_testNewFeat1_known.csv")
    NewFeat1_ReadjustKnown$NewFeature1 <- NULL
    NewFeat1_ReadjustKnown <- addNewFeature1(NewFeat1_ReadjustKnown)
    write.csv(NewFeat1_ReadjustKnown, file = "./data_manimp_tests/manImp_testNewFeat1Readjust_known.csv", row.names=FALSE)
    
    NewFeat1_ReadjustUnknown <- read.csv("./data_manimp_tests/manImp_testNewFeat1_unknown.csv")
    NewFeat1_ReadjustUnknown$NewFeature1 <- NULL
    NewFeat1_ReadjustUnknown <- addNewFeature1(NewFeat1_ReadjustUnknown)
    write.csv(NewFeat1_ReadjustUnknown, file = "./data_manimp_tests/manImp_testNewFeat1Readjust_unknown.csv", row.names=FALSE)
    
    NewFeat1_ReadjustHoldout <- read.csv("./data_manimp_tests/manImp_testNewFeat1_holdout.csv")
    NewFeat1_ReadjustHoldout$NewFeature1 <- NULL
    NewFeat1_ReadjustHoldout <- addNewFeature1(NewFeat1_ReadjustHoldout)
    write.csv(NewFeat1_ReadjustHoldout, file = "./data_manimp_tests/manImp_testNewFeat1Readjust_holdout.csv", row.names=FALSE) 
    
    NewFeat2_ReadjustKnown <- read.csv("./data_manimp_tests/manImp_testNewFeat2_known.csv")
    NewFeat2_ReadjustKnown$NewFeature2 <- NULL
    NewFeat2_ReadjustKnown <- addNewFeature2(NewFeat2_ReadjustKnown)
    write.csv(NewFeat2_ReadjustKnown, file = "./data_manimp_tests/manImp_testNewFeat2Readjust_known.csv", row.names=FALSE)
    
    NewFeat2_ReadjustUnknown <- read.csv("./data_manimp_tests/manImp_testNewFeat2_unknown.csv")
    NewFeat2_ReadjustUnknown$NewFeature2 <- NULL
    NewFeat2_ReadjustUnknown <- addNewFeature2(NewFeat2_ReadjustUnknown)
    write.csv(NewFeat2_ReadjustUnknown, file = "./data_manimp_tests/manImp_testNewFeat2Readjust_unknown.csv", row.names=FALSE)
    
    NewFeat2_ReadjustHoldout <- read.csv("./data_manimp_tests/manImp_testNewFeat2_holdout.csv")
    NewFeat2_ReadjustHoldout$NewFeature2 <- NULL
    NewFeat2_ReadjustHoldout <- addNewFeature2(NewFeat2_ReadjustHoldout)
    write.csv(NewFeat2_ReadjustHoldout, file = "./data_manimp_tests/manImp_testNewFeat2Readjust_holdout.csv", row.names=FALSE) 
      
    NewFeat1NewFeat2_ReadjustAfterKnown <- read.csv("./data_manimp_tests/manImp_testNewFeat1NewFeat2_known.csv")
    NewFeat1NewFeat2_ReadjustAfterKnown$NewFeature1 <- NULL
    NewFeat1NewFeat2_ReadjustAfterKnown$NewFeature2 <- NULL
    NewFeat1NewFeat2_ReadjustAfterKnown <- addNewFeature1(NewFeat1NewFeat2_ReadjustAfterKnown)
    NewFeat1NewFeat2_ReadjustAfterKnown <- addNewFeature2(NewFeat1NewFeat2_ReadjustAfterKnown)
    write.csv(NewFeat1NewFeat2_ReadjustAfterKnown, file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2Readjust_known.csv", row.names=FALSE)
    
    NewFeat1NewFeat2_ReadjustAfterUnknown <- read.csv("./data_manimp_tests/manImp_testNewFeat1NewFeat2_unknown.csv")
    NewFeat1NewFeat2_ReadjustAfterUnknown$NewFeature1 <- NULL
    NewFeat1NewFeat2_ReadjustAfterUnknown$NewFeature2 <- NULL
    NewFeat1NewFeat2_ReadjustAfterUnknown <- addNewFeature1(NewFeat1NewFeat2_ReadjustAfterUnknown)
    NewFeat1NewFeat2_ReadjustAfterUnknown <- addNewFeature2(NewFeat1NewFeat2_ReadjustAfterUnknown)
    write.csv(NewFeat1NewFeat2_ReadjustAfterUnknown, file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2Readjust_unknown.csv", row.names=FALSE)
      
    NewFeat1NewFeat2_ReadjustAfterHoldout <- read.csv("./data_manimp_tests/manImp_testNewFeat1NewFeat2_holdout.csv")
    NewFeat1NewFeat2_ReadjustAfterHoldout$NewFeature1 <- NULL
    NewFeat1NewFeat2_ReadjustAfterHoldout$NewFeature2 <- NULL
    NewFeat1NewFeat2_ReadjustAfterHoldout <- addNewFeature1(NewFeat1NewFeat2_ReadjustAfterHoldout)
    NewFeat1NewFeat2_ReadjustAfterHoldout <- addNewFeature2(NewFeat1NewFeat2_ReadjustAfterHoldout)
    write.csv(NewFeat1NewFeat2_ReadjustAfterHoldout, file = "./data_manimp_tests/manImp_testNewFeat1NewFeat2Readjust_holdout.csv", row.names=FALSE)
    
    manualset <- creditset
    manualset <- treatI01(manualset)
    manualset <- treatS01(manualset)
    manualset <- treatS02(manualset)
    manualset <- addNewFeature1(manualset)
    manualset <- addNewFeature2_original(manualset)
    manualset <- asFactorTimeEmp(manualset)
    set.seed(23536)
    testAllManImpOriginal <- missForest(manualset,
                                        variablewise = TRUE,
                                        replace = TRUE,
                                        verbose=TRUE,
                                        maxiter=maxiter,
                                        ntree=ntree,
                                        parallelize = "forests")
    write.csv(testAllManImpOriginal$ximp[INDICES_KNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testAllManImpOriginal_known.csv", row.names=FALSE)
    write.csv(testAllManImpOriginal$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "./data_manimp_tests/manImp_testAllManImpOriginal_unknown.csv", row.names=FALSE)
    write.csv(testAllManImpOriginal$ximp[INDICES_HOLDOUT_DATA,],  file = "./data_manimp_tests/manImp_testAllManImpOriginal_holdout.csv", row.names=FALSE)
  }
  
  #####################################
  # Put into effect
  #####################################
  
  # treatI01, treatS01, treatS02, addNewFeature1, addNewFeature2, asFactorTimeEmp
  
  manualset <- creditset
  manualset <- treatI01(manualset)
  manualset <- treatS01(manualset)
  manualset <- treatS02(manualset)
  manualset <- addNewFeature1(manualset)
  manualset <- addNewFeature2(manualset)
  manualset <- asFactorTimeEmp(manualset)
  
  #####################################
  # Imputing the rest with missForest
  #####################################
  
  print(noquote("Running final imputation. Will overwrite files."))
  
  maxiter <- 20
  ntree <- 100
  
  
  set.seed(23536)
  
  imputed_final <- missForest(manualset,
                              variablewise = TRUE,
                              replace = TRUE,
                              verbose=TRUE,
                              maxiter=maxiter,
                              ntree=ntree,
                              parallelize = "forests")
  
  #################
  # Save
  #################
  
  write.csv(imputed_final$ximp, file = "dataset_imputed.csv")
  
  write.csv(imputed_final$ximp[INDICES_KNOWN_OUTCOME,],  file = "dataset_manually_imputed_known.csv", row.names=FALSE)
  write.csv(imputed_final$ximp[INDICES_UNKNOWN_OUTCOME,],  file = "dataset_manually_imputed_unknown.csv", row.names=FALSE)
  write.csv(imputed_final$ximp[INDICES_HOLDOUT_DATA,],  file = "dataset_manually_imputed_holdout.csv", row.names=FALSE)
  
  remove(maxiter, ntree)
  
}

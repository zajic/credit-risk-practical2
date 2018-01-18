creditset <- read.csv("dataset_manually_imputed_known2.csv")

temp <- read.csv("dataset_manually_imputed_known.csv")

creditset$GOOD <- temp$GOOD

creditset$NewFeature1 <- (creditset$D_02) / (creditset$I_02)
creditset$NewFeature2 <- (creditset$D_02) - (creditset$I_02)

write.csv(creditset, file = "dataset_manually_imputed_known2.csv")

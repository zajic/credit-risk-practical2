creditset <- read.table("csv_files/dataset_imputed_known.csv", na.strings = ".", sep = ",", header = T)
attach(creditset)


#logistic reg
logistic <- glm(GOOD ~ as.factor(occ_code) + as.factor(time_emp) + as.factor(res_indicator) +
                  cust_age + as.factor(CA_01) + CA_02 + as.factor(CA_03) + ER_01 + ER_02 + 
                  as.factor(S_01) + as.factor(S_02) + disp_income + as.factor(I_01) + 
                  I_02 + I_03 + I_04 + D_01 + D_02 + I_05 + I_06 + P_01,
                  family=binomial(link='logit'),data=creditset)


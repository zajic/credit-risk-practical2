
library(h2o)

h2o.init(max_mem_size = "50g", nthreads = -1)

data = read.table("dataset_modelling.csv", sep=",", header=T,na.strings=".")

myData = h2o.importFile("data_holdout_imputed/dataset_imputed_known.csv", sep=",",header=T)

missingResponses <- which(is.na(data$GOOD))

myData <- myData[-missingResponses,]

x<-floor(nrow(myData)*0.7)

splits <- h2o.splitFrame(myData, c(0.8,0.1), seed=1234)

trainset <- h2o.assign(splits[[1]],'train.hex')

validset <- h2o.assign(splits[[2]],'valid.hex')

testset <- h2o.assign(splits[[3]],'test.hex')

response <- "GOOD"

trainset$GOOD <- as.factor(trainset$GOOD)

predictors <- setdiff(names(trainset),response)

hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(23),c(23,23),c(23,23,23),c(23,23,23,23),c(40),c(40,40),c(40,40,40),c(40,20,10),c(100,100,100),c(100),c(100,100),c(200,100,50),c(50,100,200)),
  input_dropout_ratio=c(0,0.05,0.01),
  rate=c(0.01,0.02,0.03),
  rate_annealing=c(1e-8,1e-7)
)

search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=10, stopping_tolerance=1e-2)

dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=trainset,
  validation_frame=validset, 
  x=predictors, 
  y=response,
  epochs=200,
  stopping_metric="misclassification",
  stopping_tolerance=1e-2,        
  stopping_rounds=50,
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  variable_importances=T,
  momentum_start=0.5,             ## manually tuned momentum
  momentum_stable=0.9, 
  momentum_ramp=1e7, 
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params=hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="err",decreasing=FALSE)

best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss

pred <- h2o.predict(best_model,testset)

p<-h2o.performance(best_model, testset)

summary(best_model)

head(as.data.frame(h2o.varimp(best_model)))

plot(p)

p


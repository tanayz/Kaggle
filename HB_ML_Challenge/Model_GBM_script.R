
## ##########################################################
##
## GBM Model  5/27/14
## Jeff Hebert
##
## ##########################################################

#setwd("C:\\Copy tanayz@outlook.com\\Data\\Kaggle\\HB_ML_Challenge\\")
setwd("/home/tanay/Copy/Data/Kaggle/HB_ML_Challenge")
# Read Data and prep for GBM

train = read.csv("training.csv")
test = read.csv("test.csv")

# Save response vectors to use later
train.y = train[,32:33]


# Change -999 to NA so GBM will use algorithm for sparse data
train[train==-999] <- NA
test[test==-999] <- NA

# Change Label to 0,1 for GBM model. b = 0, s = 1
train$Label=as.numeric(train$Label)-1


# Train Model
library(gbm)
set.seed(100)
gbmModel = gbm(Label~.-Weight, data=train,n.trees = 150,shrinkage = 0.1, 
               cv.folds = 10,                   # 5 fold cross validation
               weights=train$Weight,           # Set observation weights 
               interaction.depth=3,            # Set interaction depth
               verbose=TRUE)                   # use verbose response to see model training
#########################################################################
# fitControl <- trainControl(## 10-fold CV
#   method = "repeatedcv",
#   number = 10,
#   ## repeated ten times
#   repeats = 10)
# gbmfit1 <- train(Label~.-Weight, data=train,method = "gbm",
#                  trControl = fitControl,preProc = c("center", "scale"))
# gbmfit1
#########################################################################
# Get predictions and determine cutoff threshold using pROC
gbmPrediction = predict(gbmModel, newdata=train, n.trees=gbmModel$n.trees, type="response")
#The final values used for the model were n.trees = 150, interaction.depth = 3 and shrinkage
#= 0.1
# gbmPrediction = predict(gbmfit1, newdata=train, type="prob")
# library(pROC)
# auc = roc(train$Label, gbmPrediction)
# plot(auc, print.thres=TRUE)

# Threshold to set results
threshold = 0.002

table(train$Label,gbmPrediction>=threshold)

#     FALSE   TRUE
#  0 140711  23622
#  1  39574  46093

# This code creates a field for predicted response so you can calculate AMS
predicted=rep("b",250000)
predicted[gbmPrediction>=threshold]="s"
AMS(pred=predicted,real=train.y$Label,weight=train$Weight)

# [1] 2.001005


# Make predictins on test set and create submission file
gbmTestPrediction = predict(gbmModel, newdata=test, n.trees=gbmModel$n.trees, type="response")

#gbmTestPrediction = predict(gbmfit1, newdata=test, n.trees=gbmModel$n.trees, type="response")


predicted=rep("b",550000)
predicted[gbmTestPrediction>=threshold]="s"
weightRank = rank(gbmTestPrediction,ties.method= "random")


submission = data.frame(EventId = test$EventId, RankOrder = weightRank, Class = predicted)
write.csv(submission, "submission.csv", row.names=FALSE)


# This submission scored 2.00158 which is quite close to the test response. 


#################


## Function to calculate AMS from predictions
# Modified from TomHall's code to use s and b instead of 0 and 1
# https://www.kaggle.com/c/higgs-boson/forums/t/8216/r-code-for-ams-metric

AMS = function(pred,real,weight)
{
  #a = table(pred,real)
  pred_s_ind = which(pred=="s")                          # Index of s in prediction
  real_s_ind = which(real=="s")                          # Index of s in actual
  real_b_ind = which(real=="b")                          # Index of b in actual
  s = sum(weight[intersect(pred_s_ind,real_s_ind)])      # True positive rate
  b = sum(weight[intersect(pred_s_ind,real_b_ind)])      # False positive rate
  
  b_tau = 10                                             # Regulator weight
  ans = sqrt(2*((s+b+b_tau)*log(1+s/(b+b_tau))-s))
  return(ans)
}



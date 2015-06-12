setwd("E:\\Copy\\Data\\Kaggle\\HB_ML_Challenge\\xgboostR\\demo\\kaggle-higgs")
setwd("/mnt/hgfs/E/Copy/Data/Kaggle/HB_ML_Challenge/xgboostR/demo/kaggle-higgs")
install.packages("doMC", repos="http://R-Forge.R-project.org") 
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(doMC)
library(foreach)
load("rffit.RData")
registerDoMC(cores=8)
imp <- as.data.frame(rffit$importance[order(rffit$importance),])
barplot(t(imp), col='blue')
dtrain <- read.csv("data/training.csv", header=TRUE)
#dt <- read.csv("data/training.csv", header=TRUE)
dtrain[33] <- dtrain[33] == "s"
label <- as.numeric(dtrain[[33]])
data <- as.matrix(dtrain[2:31])
testsize <- 550000
weight <- as.numeric(dtrain[[32]]) * testsize / length(label)
y=label
dat=dtrain[2:31][,rownames(rffit$importance)[order(-rffit$importance)][1:24]]
sigDist <- sigest(y ~ as.matrix(dat), data=dat, frac = 1)
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-20:100))
svmFit <- train(dat,y,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "cv", number = 86, classProbs =  TRUE))
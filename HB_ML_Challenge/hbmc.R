setwd("/home/tanay/Copy/Data/Kaggle/HB_ML_Challenge")
train = read.csv("training.csv",header=T)
test = read.csv("test/test.csv",header=T)
nzv <- nearZeroVar(train, saveMetrics = TRUE) #Checking near zero values
nzv[nzv$nzv, ][1:6, ]
table(train$Label) # Checking class distribution
#descrCor <- cor(train)
#highCorr <- sum(abs(train[upper.tri(train)]) > 0.999)
head(train)

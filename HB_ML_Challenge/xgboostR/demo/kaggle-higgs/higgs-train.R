# include xgboost library, must set chdir=TRURE
#setwd("/mnt/hgfs/C/Copy tanayz@outlook.com/Data/Kaggle/HB_ML_Challenge/xgboostR/demo/kaggle-higgs")
setwd("/home/tanay/Copy/Data/Kaggle/HB_ML_Challenge/xgboostR/demo/kaggle-higgs")
source("../../wrapper/xgboost.R", chdir=TRUE)
#setwd("E:\\Copy\\Data\\Kaggle\\HB_ML_Challenge\\xgboostR\\demo\\kaggle-higgs")
testsize <- 550000


dtrain <- read.csv("training-public.csv", header=TRUE)
dt <- read.csv("data/training.csv", header=TRUE)
dtrain[33] <- dtrain[33] == "s"
label <- as.numeric(dtrain[[33]])
data <- as.matrix(dtrain[2:31])
# data <- as.matrix(x1)
weight <- as.numeric(dtrain[[32]]) * testsize / length(label)

sumwpos <- sum(weight * (label==1.0))
sumwneg <- sum(weight * (label==0.0))
print(paste("weight statistics: wpos=", sumwpos, "wneg=", sumwneg, "ratio=", sumwneg / sumwpos))

xgmat <- xgb.DMatrix(data, info = list(label=label, weight=weight), missing = -999.0)
param <- list("objective" = "binary:logitraw",
              "scale_pos_weight" = sumwneg / sumwpos,
              "bst:eta" = 0.1,
              "bst:max_depth" = 6,
              "eval_metric" = "auc",
              "eval_metric" = "ams@0.15",
              "silent" = 1,
              "nthread" = 16)
watchlist <- list("train" = xgmat)
nround = 220
print ("loading data end, start to boost trees")
bst = xgb.train(param, xgmat, nround, watchlist );
# save out model
xgb.save(bst, "higgs.model")
print ('finish training')
############################################################################################
# xvtr=dtrain$DER_mass_MMC*dtrain$DER_pt_ratio_lep_tau/(dtrain$DER_sum_pt+0.0000001)
# xvts=dtest$DER_mass_MMC*dtest$DER_pt_ratio_lep_tau/(dtest$DER_sum_pt+0.0000001)
x<-dtrain[2:31]
# x1=data.frame(x,xvtr)
# nnetAv <- avNNet(x, dtrain[[33]],size = 5,decay = 0.01,repeats = 5,linout = TRUE,
#                  trace = FALSE,maxit = 500,verbose=T)
# rffit <- train(x, label,method = "RRF",preProc = c("center", "scale"),
#                trControl = trainControl(method = "cv"))
# larft <- train(x,label,method = "lars2",preProc = c("center", "scale"),
#                trControl = trainControl(method = "cv"))
# fbft <- train(x,label,method = "foba",preProc = c("center", "scale"),
#               trControl = trainControl(method = "cv"))
# svmft <- tune.svm(x,label, sampling = "fix",gamma = 2^c(-8,-4,0,4), 
#                   cost = 2^c(-8,-4,-2,0))
# deft <- train(x,label,method = "DENFIS",preProc = c("center", "scale"),
#               trControl = trainControl(method = "cv"))
btft <- train(x,label,method = "Boruta",preProc = c("center", "scale"),
              trControl = trainControl(method = "cv"))

ypre0= xgb.predict(bst, xgmat)  #AMS : 4.33
ypre1=predict(nnetAv, newdata=x)#AMS : 2.94
ypre2=predict(larft, newdata=x) #AMS : 1.55
ypre3=predict(fbft, newdata=x)  #AMS : 1.56
ypre4=predict(rffit,newdata=x)  #AMS : 35.39
ypre6=predict(grft,newdata=x)   #AMS : 2.87
ypre = (ypre0*2.3+ypre1*1.5+ypre6+ypre4*0.10) 
rorder <- rank(ypre0, ties.method="first")

threshold <- 0.15
# to be completed
ntop <- length(rorder) - as.integer(threshold*length(rorder))
tlabel <- ifelse(rorder > ntop, "s", "b")

AMS(tlabel,dt$Label,dt$Weight)

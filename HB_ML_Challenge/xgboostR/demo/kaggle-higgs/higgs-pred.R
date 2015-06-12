# include xgboost library, must set chdir=TRURE
source("../../wrapper/xgboost.R", chdir=TRUE)

modelfile <- "higgs.model"
outfile <- "higgs.pred.csv"
dtest <- read.csv("test-public.csv", header=TRUE)
#x2=data.frame(dtest[2:31],xvts)
data <- as.matrix(dtest[2:31])
data <- as.matrix(x2)
idx <- dtest[[1]]

xgmats <- xgb.DMatrix(data, missing = -999.0)
bst <- xgb.Booster(params=list("nthread"=16), modelfile=modelfile)
ypred0 <- xgb.predict(bst, xgmats)
#######################################new model##################
ypred1=predict(nnetAv, newdata=dtest[2:31])
ypred2=predict(larft, newdata=dtest[2:31])
ypred3=predict(fbft, newdata=dtest[2:31])
ypred4=predict(rffit, newdata=dtest[2:31])
ypred6=predict(grft, newdata=dtest[2:31])
          

###########################################
ypred=(ypred0*2.3+ypred1*1.5+ypred6+ypred4*0.10)
#ypred=ypred0+0.7*ypred4
rorder <- rank(ypred, ties.method="first")

threshold <- 0.15
# to be completed
ntop <- length(rorder) - as.integer(threshold*length(rorder))
plabel <- ifelse(rorder > ntop, "s", "b")
#####################AMS################################

###########################################################
outdata <- list("EventId" = idx,
                "RankOrder" = rorder,
                "Class" = plabel)
write.csv(outdata, file = outfile, quote=FALSE, row.names=FALSE)
####################AMS Calculation###################
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


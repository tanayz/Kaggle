WeightedGini <- function(solution, weights, submission){
  df = data.frame(solution, weights, submission)
  n <- nrow(df)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = cumsum(df$weights/sum(df$weights))
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / df$cumPosFound[n]
  sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
}

NormalizedWeightedGini <- function(solution, weights, submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}

pred0 <- predict(fbfit, newdata=(ntrain[1:87]),fbfit$bestTune)

#pred0 <- predict(fbfit4, newdata=(ntrain[c('var8', 'var10','var13', 'var15')]))
pred0 <- predict(lpfit, newdata=trainx)

WeightedGini(solution=y,weights=ntrain$var11,submission=p0)

NormalizedWeightedGini(solution=ntrain$target,weights=ntrain$var11,submission=pred)


##To check #cubist = Cubist,icr = fastICA,krlsPoly = KRLS@@,krlsRadial=KRLS, kernlab,lars2=lars,lasso = elasticnet
           #leapBackward = leaps,leapForward = leaps,leapSeq=leaps,lmStepAIC = MASS,M5=RWeka,M5Rules = RWeka
            #neuralnet=neuralnet,pcr=pls,qrf = quantregForest,relaxo = relaxo, plyr,ridge=elasticnet,
            #rlm=MASS,rvmLinear=kernlab,rvmPoly=kernlab,rvmRadial=kernlab
##Use one regression inside another
##Try to boost further

#bmnfit = 0.2737707     (4)(p1)
#fbfit = 0.3554484   ** (All)(p5) #0.31742
#gbm2 = 0.2880          (4)(p10)
#glm1 = 0.2737673       (4)
#lmafit = 0.3468909  ** (All)(p6)
#nnetAvg = 0.3100403 ** (4)(p3)
#prTune = 0.2880607     (p2)
#nntfit =  0.2864658    (p7)
#penfit = 0.2737994     (4)
#spcTune = 0.2619298    (4)
#nnetAvgA = 0.4445072   (All) (p13)
#lbfit = 0.2750875      (All)
#trefit = 0.1873688     (All)######
#ficfit = 0.03159743         ######
#lasfit = 0.2836739     (All)(p4)
#lasfit4 = 0.2730634    (4)
#larfit = 0.3447516     (p11)




trainx=ntrain[,c('var13','var15','var8','weatherVar108','var4','var17','geodemVar24','geodemVar37','weatherVar199','var10','weatherVar203',
'weatherVar186','var3','var16','weatherVar48','weatherVar214','weatherVar224','weatherVar153','weatherVar148','geodemVar30',
'geodemVar36','var14','var9','weatherVar184','weatherVar232','weatherVar104','weatherVar182','weatherVar70','weatherVar48',
'weatherVar184','weatherVar133','weatherVar170','weatherVar203','weatherVar203','geodemVar31','weatherVar142')]

testx=ntest[,c('var13','var15','var8','weatherVar108','var4','var17','geodemVar24','geodemVar37','weatherVar199','var10','weatherVar203',
               'weatherVar186','var3','var16','weatherVar48','weatherVar214','weatherVar224','weatherVar153','weatherVar148','geodemVar30',
               'geodemVar36','var14','var9','weatherVar184','weatherVar232','weatherVar104','weatherVar182','weatherVar70','weatherVar48',
               'weatherVar184','weatherVar133','weatherVar170','weatherVar203','weatherVar203','geodemVar31','weatherVar142')]

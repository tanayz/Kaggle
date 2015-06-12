plft <- train(x,label,method = "pls",preProc = c("center", "scale"),trControl = trainControl(method = "cv"))
varimp = varImp(plft, scale=TRUE)
head(x[,rownames(varimp$importance)[order(-varimp$importance$Overall)][1:24]])
x1=x[,rownames(varimp$importance)[order(-varimp$importance$Overall)][1:24]]

x2=dtest[2:31][,rownames(varimp$importance)[order(-varimp$importance$Overall)][1:24]]

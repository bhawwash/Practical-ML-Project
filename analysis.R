library(caret)
library(parallel)
library(doParallel)
#load raw data
cl=makeCluster(detectCores())
registerDoParallel(cl)

training = read.csv("pml-training.csv")
testing = read.csv("pml-testing.csv")

training = training[,-c(1:7)]
training = training[,c(153,1:152)]
training[,2:153] =as.data.frame(sapply(training[,-1],function(x){as.numeric(as.character(x))}))
PercNull<- colSums(1*is.na(training))/dim(training)[1]
training = training[,PercNull<0.9]

testing = testing[,-c(1:7)]
testing = testing[,c(153,1:152)]
testing[,2:153] =as.data.frame(sapply(testing[,-1],function(x){as.numeric(as.character(x))}))
testing = testing[,PercNull<0.9]



set.seed(1234)
IDX = createDataPartition(y=training$classe,p=0.7,list=F)
train_set = training[IDX,]
valid_set = training[-IDX,]


pca_processing = preProcess(train_set[,-1],method="pca",pcaComp=10)
m=train(train_set[,1]~.,data=predict(pca_processing,train_set[,-1]),method="lda")



d = preProcess(train_set[,-1],method=c("center","scale"))
train_set[,2:53] = predict(d,newdata=train_set[,-1])

valid_set[,2:53] = predict(d,newdata=valid_set[,-1])

testing[,2:53]= predict(d,newdata=testing[,-1])

control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(123)
m_lda = train(classe~.,data=train_set,method="lda",trControl=control)
m_gbm = train(classe~.,data=train_set,method="gbm",trControl=control)
m_rf = train(classe~.,data=train_set,method="rf",trControl=control)

result=resamples(list(LDA=m_lda,RF=m_rf,GBM=m_gbm))
bwplot(result)

rf_accuracy<- predict(m_rf, valid_set)
print(confusionMatrix(rf_accuracy, valid_set$classe))

predict(m_rf,testing)


"B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B" 

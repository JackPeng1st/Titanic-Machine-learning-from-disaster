library(DMwR)
library(dplyr)
library(randomForest)
library(mice)
train=read.csv("titanic.train.csv",sep=",",na.strings = "", stringsAsFactors=FALSE)
test=read.csv("titanic.test.csv",sep=",",na.strings = "", stringsAsFactors=FALSE)
#which(colSums(sapply(train, is.na))==F)
sapply(train,function(x) sum(is.na(x)))
sapply(test,function(x) sum(is.na(x)))
#用集中趨勢填NA
train.1=centralImputation(train)
test.1=centralImputation(test)
sapply(train.1,function(x) sum(is.na(x)))
sapply(test.1,function(x) sum(is.na(x)))
#Name,Ticket,Cabin have too many categories(建成factor會有太多類別，因為random forest 最多容許53格類別 ) ,so 刪除他們
all.1=rbind(within(train.1,rm("Survived")),test.1)
all.1=subset(all.1,select=-c(Name,Ticket,Cabin))

#需要將CHAR變數轉factor(因為要用randomforest)
chr.var=all.1[,(sapply(all.1,is.character))]
num.var=all.1[,sapply(all.1,is.numeric)]

fac.var=sapply(chr.var,as.factor)
all.1=data.frame(fac.var,num.var)

train.1=all.1[1:nrow(train),]
test.1=all.1[(nrow(train)+1):nrow(all.1),]
Survived=train$Survived
train.1=data.frame(train.1,Survived)
sapply(train.1,function(x) sum(is.na(x)))
sapply(train.1,function(x) sum(is.na(x)))

model.rf=randomForest(Survived~.,data=train.1,ntree=1000,proximity=TRUE)

prediction.rf=predict(model.rf,test.1)
prediction.rf[prediction.rf>0.65]=1
prediction.rf[prediction.rf<=0.65]=0
result.rf=cbind(PassengerId=test$PassengerId,Survived=prediction.rf)

write.csv(result.rf,"rf.csv",row.names=F)

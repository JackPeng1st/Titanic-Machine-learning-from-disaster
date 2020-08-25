library(ggplot2)
library(DMwR)
library(dplyr)
library(randomForest)
library(mice)
library(rpart)

titanic_train=read.csv("C:/Users/jackp/OneDrive/文件/專題/titanic/train.csv",sep=",",na.strings = "")
titanic_test=read.csv("C:/Users/jackp/OneDrive/文件/專題/titanic/test.csv",sep=",",na.strings = "")

sapply(titanic_train,function(x) sum(is.na(x)))
sapply(titanic_test,function(x) sum(is.na(x)))

titanic_train=subset(titanic_train,select=-Cabin)
titanic_test=subset(titanic_test,select=-Cabin)

#統計數字
data_all=rbind(subset(titanic_train,select=-Survived),titanic_test)
sapply(data_all,function(x) sum(is.na(x)))
ggplot(data_all$Sex)
#ggplot(data_all, aes(x=Sex))+ geom_bar()
ggplot(data_all, aes(x=Pclass,fill=Sex))+ geom_bar()
ggplot(data_all, aes(x=Age))+ geom_histogram()

summary(data_all)

#計算男女存活率(trainning data)
female_train=subset(titanic_train,titanic_train$Sex=="female")
male_train=subset(titanic_train,titanic_train$Sex=="male")
male_survived=0
female_survived=0
for(i in 1:577){
  if(male_train$Survived[i]==1){
    male_survived=male_survived+1
     }
}
for(i in 1:314){
  if(female_train$Survived[i]==1){
    female_survived=female_survived+1
  }
}
male_survival_rate=male_survived/length(male_train$Survived)
female_survival_rate=female_survived/length(female_train$Survived)
survival_rate=rbind(male_survival_rate,female_survival_rate)
colnames(survival_rate)="SurvivalRate"
survival_rate=as.data.frame(survival_rate)
Sex=c("man","woman")
survival_rate=data.frame(survival_rate,Sex)
ggplot(survival_rate,aes(y=SurvivalRate,x=Sex))+geom_bar(stat="identity")

##### 資料預處理
train=read.csv("C:/Users/jackp/OneDrive/文件/專題/titanic/train.csv",sep=",",na.strings = "", stringsAsFactors=FALSE)
test=read.csv("C:/Users/jackp/OneDrive/文件/專題/titanic/test.csv",sep=",",na.strings = "", stringsAsFactors=FALSE)
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

#### Randomforest
model.rf=randomForest(Survived~.,data=train.1,ntree=1000,proximity=TRUE)

prediction.rf=predict(model.rf,test.1)
prediction.rf[prediction.rf>0.65]=1
prediction.rf[prediction.rf<=0.65]=0
result.rf=cbind(PassengerId=test$PassengerId,Survived=prediction.rf)

write.csv(result.rf,"rf.csv",row.names=F)
#### Decision Tree
model.tree=rpart(Survived~.,data=train.1,,method="anova")
prediction.tree=predict(model.tree,test.1)
prediction.tree[prediction.tree>0.65]=1
prediction.tree[prediction.tree<=0.65]=0
result.tree=cbind(PassengerId=test$PassengerId,Survived=prediction.tree)

write.csv(result.rf,"tree.csv",row.names=F)

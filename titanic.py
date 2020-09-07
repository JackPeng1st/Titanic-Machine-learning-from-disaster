import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, cross_validation

#with open('C:/Users/jackp/OneDrive/文件/專題/titanic/train.csv', newline='') as csvfile:
   # rows = csv.reader(csvfile)
    #for row in rows:
       # print(row)
train_data = pd.read_csv('C:/Users/jackp/OneDrive/文件/專題/titanic/train.csv',engine='python')
test_data = pd.read_csv('C:/Users/jackp/OneDrive/文件/專題/titanic/test.csv',engine='python')

train_data.head()
test_data.head()
print(train_data.describe())
print(train_data.isnull().sum())  
print(test_data.isnull().sum())

#survival number
labels='Survived','dead'
size=(len(train_data[train_data['Survived']==1]),len(train_data[train_data['Survived']==0]))
plt.pie(size , labels = labels,autopct='%1.1f%%')
plt.axis('equal')
plt.show()

#women survival rate (train data)
women = train_data.loc[train_data.Sex == 'female']["Survived"]
survival_rate_women = sum(women)/len(women)
print(survival_rate_women)

men=train_data.loc[train_data.Sex=='male']["Survived"]
survival_rate_men=sum(men)/len(men)
print(survival_rate_men,survival_rate_women)

##bar chart
x = ['Man survival rate','Woman survival rate']
y=[survival_rate_men,survival_rate_women]
plt.bar(x, y)
plt.grid(True)
plt.show()
####
def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

train_test_data = [train_data, test_data] # combining train and test dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
print(train_data['Title'])

bar_chart('Title')
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Parch')

#####feature engineering


###將test_data和train_data合併一起處理
data_all=train_data.append(test_data)
# 將SibSp與Parch欄合併, 再刪他們
data_all['Family']=data_all['SibSp']+data_all['Parch']
data_all.drop('SibSp',1,inplace=True)
data_all.drop('Parch',1,inplace=True)

data_all.info()
data_all.reset_index(inplace=True, drop=True)
###用平均填補Age的NAs
data_all['Age']=data_all['Age'].fillna(data_all['Age'].median())

##看embarked類別裡的數量
sns.countplot(data_all['Embarked'],hue=data_all['Survived'])
data_all['Embarked']=data_all['Embarked'].fillna('S')

data_all['Fare']=data_all['Fare'].fillna(data_all['Fare'].mean())

data_all['Cabin']=data_all['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin') 

data_all.drop('Name',1,inplace=True)
data_all.drop('PassengerId',1,inplace=True)
data_all.drop('Ticket',1,inplace=True)

data_all.info()
###將非類別型態的變數轉為數值型
sns.countplot(data_all['Embarked'])
data_all['Embarked']= data_all['Embarked'].astype('category').cat.codes
data_all['Sex']= data_all['Sex'].map({'male':1, 'female':2})

sns.countplot(data_all['Cabin'])
data_all['Cabin']=data_all['Cabin'].astype('category').cat.codes
data_all.info()
####建model
data_all.drop('Survived',1,inplace=True)
X_train=data_all[0:len(train_data)]
X_test=data_all.iloc[len(train_data):]
# Scikit-learn 需要train dataset及label dataset（即答案）各一
Y_label= train_data.Survived 

#Y_pred = cross_validation.cross_val_predict(RandomForestClassifier(n_estimators=100), X_train, Y_label, cv=10)
#acc_random_forest = metrics.accuracy_score(Y_label, Y_pred)
#print (metrics.classification_report(Y_label, Y_pred) )

rf=RandomForestClassifier(criterion='gini',n_estimators=1000, min_samples_split=12,min_samples_leaf=1,oob_score=True,random_state=1,n_jobs=-1)

rf.fit(X_train,Y_label)

prediction=rf.predict(X_test)

submission=pd.DataFrame({"PassengerId":test_data['PassengerId'],"Survived":prediction})
submission.to_csv('python_rf.csv',index=False)

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:58:24 2019

@author: Srujan Kachhwaha
"""
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 16)
pd.set_option('precision', 2)
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('delinquency_data.csv')

train_data= data.drop(['msisdn','daily_decr30','cnt_ma_rech30','fr_ma_rech30','sumamnt_ma_rech30','medianamnt_ma_rech30','fr_ma_rech90','cnt_da_rech30','fr_da_rech30','cnt_loans30','amnt_loans30','maxamnt_loans30','medianamnt_loans30','payback30','pcircle'],axis=1)




for i in range(len(train_data['aon'])):
     
     
     if 'UA' in train_data['aon'][i]:
         
         train_data['aon'][i]= -0.5
        
        
train_data['aon']=train_data['aon'].astype('float')



for i in range(len(train_data['daily_decr90'])):
     
     
     if 'UA' in train_data['daily_decr90'][i]:
         
         train_data['daily_decr90'][i]= -0.5
        
        
train_data['daily_decr90']=train_data['daily_decr90'].astype('float')


for i in range(len(train_data['rental90'])):
     
     
     if 'UA' in train_data['rental90'][i]:
         
         train_data['rental90'][i]= -0.5
        
        
train_data['rental90']=train_data['rental90'].astype('float')




        
        
train_data['rental90']=train_data['rental90'].astype('float')

train_data=train_data.drop(['rental30'],axis=1)

train_data=train_data.drop(['pdate'],axis=1)

"""
arrays_data=train_data.values

y_feature=arrays_data[:,1]

x_feature=arrays_data[:,2:]


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_feature,y_feature)

#gradiant decsent
"""

train_data=train_data.drop(['last_rech_date_da','medianmarechprebal30','medianamnt_ma_rech90','medianmarechprebal90','cnt_da_rech90','fr_da_rech90','amnt_loans90','medianamnt_loans90'],axis=1)


print(train_data['label'][train_data['aon']<=1825].value_counts(normalize = True)[1]*100)

#sbn.barplot(train_data['label'][train_data['aon']<=1825].value_counts(normalize = True)[1],train_data['label'][train_data['aon']<=1825].value_counts(normalize = True)[0])


print(train_data['label'][train_data['aon']>=1825].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['daily_decr90']<0].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['daily_decr90']<3645].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['daily_decr90']>3645].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['rental90']<0].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['rental90']>5764].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['last_rech_date_ma']==0].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['last_rech_date_ma']<=7].value_counts(normalize = True)[1]*100)


print(train_data['label'][train_data['last_rech_date_ma']>7].value_counts(normalize = True)[1]*100)


print(train_data['label'][train_data['last_rech_amt_ma']<770].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['last_rech_date_ma']>770].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['last_rech_date_ma']>770].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['cnt_ma_rech90']<8].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['cnt_ma_rech90']>8].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['sumamnt_ma_rech90']<223].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['sumamnt_ma_rech90']>223].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['maxamnt_loans90']==6].value_counts(normalize = True)[1]*100)

print(train_data['label'][train_data['maxamnt_loans90']>6].value_counts(normalize = True)[1]*100)


print(train_data['label'][train_data['payback90']<10].value_counts(normalize = True)[1]*100)


print(train_data['label'][train_data['payback90']>10].value_counts(normalize = True)[1]*100)


#model selecting and evaluation.

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



num_folds =10

kfold=KFold(n_splits=num_folds)

arrays_data=train_data.values

x_feature = arrays_data[:,2:]

y_feature = arrays_data[:,1]

from sklearn.linear_model import LogisticRegression
import time as tim



model = LogisticRegression()

t0 = tim.time()

result = cross_val_score(model,x_feature,y_feature,cv=kfold)

t1 = tim.time()

print("time for logistic regression in CPU: ",t1-t0)
print("Accuracy of 10 data sets:",result)

print("Accuracy of model: ", result.mean())

from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(criterion="entropy", max_depth = 4)
t0=tim.time()
result = cross_val_score(model,x_feature,y_feature,cv=kfold)

t1=tim.time()

print("Time for tree classification cpu:", t1-t0)
print("Accuracy of 10 data sets:",result)

print("Accuracy of model: ", result.mean())



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train50,x_test50,y_train50, y_test50 = train_test_split(x_feature,y_feature,test_size=0.60)


x_train20, x_test20,y_train20, y_test20=train_test_split(x_train50,y_train50, test_size=0.60)




#from sklearn.metrics import accuracy_score
from sklearn import svm
model = svm.SVC()
t0=tim.time()
model.fit(x_train20,y_train20)
t1=tim.time()
y_pred=model.predict(x_test20)
print("time for SVM 20% data", t1-t0)
print("Accuracy SVM:", accuracy_score(y_test20,y_pred)) #88%

t1=tim.time()
model.fit(x_train50,y_train50)
t2 = tim.time()

print("time for svm 50$% data: ",t2-t1)
y_pred50=model.predict(x_test50)

print("Accuracy of svm for 50%:",accuracy_score(y_test50,y_pred50)) #88%


# calculating Accuracy mannualy------>












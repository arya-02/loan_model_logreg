import pandas as pd
import numpy as np

#reading the dataset
data=pd.read_csv('loan_model.csv')
data.drop(['Loan_ID'],axis=1,inplace=True)

#Convertting the categorical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
data['Married']=le.fit_transform(data['Married'])
data['Education']=le.fit_transform(data['Education'])
data['Self_Employed']=le.fit_transform(data['Self_Employed'])
data['Property_Area']=le.fit_transform(data['Property_Area'])
data['Loan_Status']=le.fit_transform(data['Loan_Status'])

y=data['Loan_Status']
x=data.iloc[:,0:11]

#cleaning the data
x['Dependents'].fillna(0,inplace=True)
x['LoanAmount'].fillna(x['LoanAmount'].mean(),inplace=True)
x['Loan_Amount_Term'].fillna(360,inplace=True)
x['Credit_History'].fillna(1,inplace=True)
x['Dependents'].replace(to_replace='3+',value=3,inplace=True)

print(x.head(5))

#splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

#scaling the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#training and predicting
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)

#testing accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
score=accuracy_score(y_test,pred)
mat=confusion_matrix(y_test,pred)
print(score)
print(mat)

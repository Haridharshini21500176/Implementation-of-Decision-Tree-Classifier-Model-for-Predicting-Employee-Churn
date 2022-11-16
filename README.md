# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset
3.Check for any missing value in the dataset using isnull function.
4.From sklearn.tree import DecisionTreeClassifier and use criteria as entropy
5.Find the accuracy of the model and predict the required values by importing the required modules from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HariDharshini.S
RegisterNumber:212221230033  
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![198622875-45307d86-74c2-4e6c-a424-18d1a5a6bb67](https://user-images.githubusercontent.com/94168395/202194289-3e63c7c9-e8b5-435b-86ae-f96f16c8bead.png)
![198623007-6c66644a-90f8-46e1-9714-e9cae680a6eb](https://user-images.githubusercontent.com/94168395/202194386-2c99d8ba-f47a-4113-ae77-5819c52a0d7c.png)
![198623098-83e674c9-770c-40c2-b34f-578a4a43d000](https://user-images.githubusercontent.com/94168395/202194421-8bcf9849-e409-4572-9dd3-72d3e68c4395.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

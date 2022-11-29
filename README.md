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
![204431272-09725dbe-97b0-46f8-9593-573a0ef9b9c5](https://user-images.githubusercontent.com/94168395/204470461-056ed41c-e548-4115-9a95-5be2f62b20c4.png)

![204431311-c79110ea-50ec-49d7-92ca-49920f4083b1](https://user-images.githubusercontent.com/94168395/204470474-0000ce36-064d-445d-9537-108cad4cb247.png)

![204431344-c38de66d-60d2-4a67-8a62-1389a18b315b](https://user-images.githubusercontent.com/94168395/204470493-c5b977d5-d8c9-4618-b027-774824b4fb6a.png)

![204431426-4f73c285-2c96-4d04-9efc-f7edf5be0af8](https://user-images.githubusercontent.com/94168395/204470524-aba221d7-1c85-4088-a8de-55a7550caf86.png)

![204431448-c31efe7f-4f6b-4110-9fb2-cdf7d7d2785f](https://user-images.githubusercontent.com/94168395/204470538-c13eff93-7f00-4077-b234-02613e4efad0.png)

![204431514-644b2673-f477-4e43-9bc6-9b98a8ba4ffe](https://user-images.githubusercontent.com/94168395/204470549-d8b27b7d-ba45-4e64-b13a-5d3b6de7d6df.png)

![204431477-5c78000d-73c5-414a-9ee2-4e2c65d6a7a6](https://user-images.githubusercontent.com/94168395/204470615-aaf94a11-d018-4e06-b84b-928cd64d73d0.png)

![204431538-c372cce1-5b37-400d-a8e4-fa729f541267](https://user-images.githubusercontent.com/94168395/204470627-34450a30-77a5-4312-b143-279f5cb0f798.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

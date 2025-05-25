# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.

## Program:
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Manisha selvakumari.S.S.

RegisterNumber: 212223220055
*/
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![Screenshot 2025-05-25 174641](https://github.com/user-attachments/assets/fc402324-916a-4aa0-a0dc-a3e2a2f70022)

![Screenshot 2025-05-25 174703](https://github.com/user-attachments/assets/78b1d500-acf2-4667-9b6f-14538dc3c000)

![Screenshot 2025-05-25 174749](https://github.com/user-attachments/assets/39bce674-0388-450e-b7b4-bc51acc4f8f6)

![Screenshot 2025-05-25 174929](https://github.com/user-attachments/assets/bc63ef01-95d8-4229-ba89-2c552fea0562)

![Screenshot 2025-05-25 174947](https://github.com/user-attachments/assets/3fff5f87-c876-46f7-956f-9404210322f6)

![Screenshot 2025-05-25 174957](https://github.com/user-attachments/assets/ca6f6192-22fc-41e2-9a70-a9c1f09ca280)

![Screenshot 2025-05-25 175005](https://github.com/user-attachments/assets/976ace03-b173-4f27-b638-3db729898d96)

![Screenshot 2025-05-25 175016](https://github.com/user-attachments/assets/0664e3bd-ec08-4583-82b1-c61864379037)

![Screenshot 2025-05-25 175021](https://github.com/user-attachments/assets/76be94c7-f3e6-416e-ae94-ca318a619d0c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

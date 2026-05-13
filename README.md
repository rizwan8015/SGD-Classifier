# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries including pandas, sklearn, matplotlib, and seaborn.

2.Load the Iris dataset using load_iris() from sklearn.

3.Create a DataFrame with feature data and add the target column.

4.Display the first and last rows of the dataset using head() and tail().

5.Split data into features (x) and target (y).

6.Split the dataset into training and testing sets using train_test_split.

7.Train the model using SGDClassifier on the training data.

8.Make predictions on the test set with the trained model.

9.Evaluate the model using accuracy score and confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: RIZWAN B
RegisterNumber:  212224100051
*/
```
```
 import pandas as pd
      from sklearn.datasets import load_iris
      from sklearn.linear_model import SGDClassifier
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import accuracy_score,confusion_matrix
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      iris=load_iris()
      
      df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
      df['target']=iris.target
      
      print(df.head())
      print(df.tail())
      
      x=df.drop('target',axis=1)
      y=df['target']
      
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
      
      sgd=SGDClassifier(max_iter=1000,tol=1e-3)
      sgd.fit(x_train,y_train)
      y_pred=sgd.predict(x_test)
      
      acc=accuracy_score(y_test,y_pred)
      print(acc)
      cm=confusion_matrix(y_test,y_pred)
      print(cm)
```

## Output:

<img width="815" height="581" alt="image" src="https://github.com/user-attachments/assets/d1aa9991-030f-4bc8-8d24-6e97ba8e6399" />


<img width="312" height="129" alt="image" src="https://github.com/user-attachments/assets/3428ac2e-5ba9-4466-94c2-58afdaea8491" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

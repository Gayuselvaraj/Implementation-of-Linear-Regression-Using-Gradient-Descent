# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: GAYATHRI S
RegisterNumber: 212224230073 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="790" height="203" alt="Screenshot 2026-02-02 085613" src="https://github.com/user-attachments/assets/fed4bf42-912f-42b3-b275-d46ca2abd9f9" />


<img width="506" height="703" alt="Screenshot 2026-02-02 085624" src="https://github.com/user-attachments/assets/d09619f0-d7cf-4758-875d-18abb0708175" />



<img width="632" height="699" alt="Screenshot 2026-02-02 085640" src="https://github.com/user-attachments/assets/93b2e432-9106-4324-9e89-587821ff7ea2" />


<img width="874" height="97" alt="Screenshot 2026-02-02 090504" src="https://github.com/user-attachments/assets/0e66af09-44d8-4ce6-ad0e-b4e3357e4775" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

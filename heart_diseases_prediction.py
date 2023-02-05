import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data=pd.read_csv('/content/heart.csv')
heart_data.head()

heart_data.info()
heart_data.describe()

heart_data.isnull().sum()

heart_data['target'].value_counts()

x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)

model=LogisticRegression()

model.fit(x_train,y_train)

x_train_predection=model.predict(x_train)
x_train_accuracy=accuracy_score(x_train_predection,y_train)
print(x_train_accuracy)

x_test_predection=model.predict(x_test)
x_test_accuracy=accuracy_score(x_test_predection,y_test)
print(x_test_accuracy)

input_data=(67,0,0,106,223,0,1,142,0,0.3,2,2,2)
input_data_as_array=np.array(input_data)
input_data_as_array_reshape=input_data_as_array.reshape(1,-1)
prediction=model.predict(input_data_as_array_reshape)
print(prediction)

if (prediction==1):
  print("body has heart diseases")
else:
  print("body has not heart diseases")
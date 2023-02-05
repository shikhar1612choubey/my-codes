import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('/content/loan_dataset.csv')
loan_dataset.head()
# loan_dataset.shape

loan_dataset.describe()

loan_dataset.isnull().sum()

# droping missing value
loan_dataset = loan_dataset.dropna()

loan_dataset.isnull().sum()

loan_dataset.shape

# replaceing loan status from Y/N to 1/0
loan_dataset.replace({"Loan_Status": {'Y': 1, 'N': 0}}, inplace=True)
loan_dataset.head()

loan_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

loan_dataset['Dependents'].value_counts()

sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)

sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)

loan_dataset.replace(
    {'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
     'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, 'Not Graduate': 0}},
    inplace=True)

loan_dataset.head()

x = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = loan_dataset['Loan_Status']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

# svm-->support vector machine, svc-->support vector classifier
classifier = svm.SVC(kernel='linear')

classifier.fit(x_train, y_train)

x_train_prediction = classifier.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prediction, y_train)
print(x_train_accuracy)

x_test_prediction = classifier.predict(x_test)
x_test_accuracy = accuracy_score(x_test_prediction, y_test)
print(x_test_accuracy)

input_data = (1, 1, 1, 1, 0, 4583, 1508, 128, 360, 1, 0)
input_data_as_array = np.array(input_data)
input_data_as_array_reshape = input_data_as_array.reshape(1, -1)
prediction = classifier.predict(input_data_as_array_reshape)
print(prediction)

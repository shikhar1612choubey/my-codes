import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Tfidf-->term frequency inverse document frequency
# Tf(term frequency)=(number of times term x appears in a doc)/(number of terms in the document)
#idf=log(N/n),where N is the no of doc and n is the number of doc a term t appeared in.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv('/content/mail_data.csv')
raw_mail_data.head()

# if there is null data
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

mail_data.head()

mail_data.shape

# in Category column there are 2 types of mail
# label spam mail as 0; ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0  # loc-->locate

mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separting the data as texts and label
x = mail_data['Message']
y = mail_data['Category']
print(x)
print(y)

# spliting the data into traing data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x.shape, x_train.shape, x_test.shape

# feature extraction--> it is used to transform the text data into feature vectors that can be used as input to the
# Logistic regression min_df--> there should be min 1 repeated words, stop_words--> dont count 'is,are,am,the,did,
# lowercase-->convert all to lower case
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# covert y_train and y_test as integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_test_features)

# training the machine learning model
model = LogisticRegression()

model.fit(x_train_features, y_train)

# evaluating the trained model
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_score = accuracy_score(y_train, prediction_on_training_data)

print(accuracy_on_training_score)

# evaluating the trained model
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print(accuracy_on_test_data)

input_mail = [
    "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0] == 1):
    print('Ham mail')

else:
    print('Spam mail')

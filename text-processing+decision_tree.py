import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def prepocess_data(data):
    data['Sentence'] = data['Sentence'].str.strip().str.lower()
    return data


print("started.")

data = pd.read_csv("data_da3.csv")

data = prepocess_data(data)

x = data['Sentence']
y = data['Sentiment']

le = LabelEncoder()
y = le.fit_transform(y)

vect = CountVectorizer(stop_words='english')
x = vect.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=50)

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)

# Training the decision tree
clf.fit(x_train, y_train)

# Predicting the labels on the test set
y_pred = clf.predict(x_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

# importing the accuracy metric
print('Accuracy Score on train data:', accuracy_score(
    y_true=y_train, y_pred=clf.predict(x_train))*100)
print('Accuracy Score on test data:', accuracy_score(
    y_true=y_test, y_pred=clf.predict(x_test))*100)

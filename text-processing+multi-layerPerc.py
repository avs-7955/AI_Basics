import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


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

model = MLPClassifier(hidden_layer_sizes=(150, 75, 50, 25, 10),
                      activation='relu', solver='adam', max_iter=1000).fit(x_train, y_train)

y_pred = model.predict(x_test)
model.score(x_test, y_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

print('Accuracy Score on test data: ', accuracy_score(
    y_true=y_test, y_pred=y_pred)*100)

# Input 01
input1 = x_test[-1].reshape(1, -1)
print(input1)
print(model.predict(input1))

# Input 02
input2 = x_test[-2].reshape(1, -1)
print(input2)
print(model.predict(input2))

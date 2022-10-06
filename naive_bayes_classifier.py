import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Pre-processing data
def preprocess_data(data):
    data['review'] = data['review'].str.strip().str.lower()
    return data

# Importing data
data = pd.read_csv("data1.csv")
data.head()

data = preprocess_data(data)

x = data['review']
y = data['polarity']

#Vecotrizing the reviews(text) into numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()

# Splitting the training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

#Creating the model
model = MultinomialNB()
print(model.fit(x, y))
y_pred = model.predict(x_test)
model.score(x_test, y_test)

# Printing the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Checking output
print(model.predict(vec.transform(['Love this laptop simply awesome!'])))

# Function defined to predict the review's polarity
def sentiment_predictor(input):
    prediction = model.predict(vec.transform(input).toarray())
    prediction
    print(f"Input statement has {prediction[0]} sentiment.")

#Input 01
input1 = ["Worst support. It needs more improvement."]
print(f"Input 01: {input1[0]}")
sentiment_predictor(input1)

#Input 02
input2 = ["Its working not nicely."]
print(f"Input 02: {input2[0]}")
sentiment_predictor(input2)

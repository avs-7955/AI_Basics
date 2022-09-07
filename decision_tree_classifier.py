# Implement Decision tree classifier for breast cancer Wisconsin dataset
# (load_breast_cancer) and evaluate the algorithm with precision, recall sensitivity and
# specificity.
# -Tune your model further using hyper-parameters and try to come up with highest
# accuracy score.
# -Use 80 % of samples as training data size.

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# Loading the breast cancer dataset
data = load_breast_cancer()

# Showing the keys of the dataset
list(data)
print("Classes to predict:", data.target_names)
print("Name of the dataset columns:", data.feature_names)

# Extracting data attributes
X = data.data

# Extracting target/ class labels

y = data.target

print('Number of examples in the data:', X.shape)

# First two rows in the variable 'X'

print(X[:2])

# Using the train_test_split to create train and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.8)

# Importing the Decision tree classifier from the sklearn library.
#from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

# Training the decision tree classifier.

dtree.fit(X_train, y_train)

# Predicting labels on the test set.

y_pred = dtree.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(" accuracy = ", accuracy_score(y_test, y_pred))
print(" f1_score = ", f1_score(y_test, y_pred))
print("Precision = \t\t", tp/(tp+fp))
print("Recall/Sensitivity = \t", tp/(tp + fn))
print("Specifity = \t\t", tn/(tn + fp))
print("\nClassification Matrix")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

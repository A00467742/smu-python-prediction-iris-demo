from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
data = load_iris()

X = data['data']
y = data['target']

train_X, test_X,train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=44)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_y)

y_pred = clf.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)

print(accuracy)

train_pred = clf.predict(train_X)
train_accuracy = accuracy_score(train_y, train_pred)

print(train_accuracy)

print(data['target_names'][y_pred])

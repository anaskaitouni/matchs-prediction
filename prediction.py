import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matches = pd.read_csv("matches.csv", index_col=0)



matches.head()
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]


def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
clf = LogisticRegression(lr=0.01)
clf.fit(train[predictors],train["target"])
y_pred = clf.predict(test[predictors])


def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

acc = accuracy(y_pred, test["target"])
print(acc)



def make_predictions(model, data, predictors):
    if not all(item in data.columns.tolist() for item in predictors):
        raise ValueError("One or more predictors are missing in the data frame.")
    preds = model.predict(data[predictors].values)
    combined = pd.DataFrame({
        "actual": data["target"],
        "predicted": preds
    }, index=data.index)
    return combined
test_set = matches[matches["date"] >= '2022-01-01']
combined_predictions = make_predictions(clf, test_set, predictors)
combined_predictions = combined_predictions.merge(test_set[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined_predictions.head())
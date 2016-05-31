from sklearn import cross_validation
from sklearn import datasets
from sklearn import linear_model

boston = datasets.load_boston()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    boston.data, boston.target, test_size=0.4, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print regr.score(X_test, y_test)

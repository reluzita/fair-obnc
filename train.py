from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'LogReg': LogisticRegression(random_state=42),
    'DT': DecisionTreeClassifier(random_state=42)
}

def fit_predict(X_train, y_train, X_test, clf_name='LogReg'):
    """
    Fit a classifier and predict on the test set

    Parameters
    ----------
    X_train : pandas.DataFrame
        Dataframe containing the features of the training set
    y_train : pandas.Series
        Series containing the labels of the training set
    X_test : pandas.DataFrame
        Dataframe containing the features of the test set
    clf_name : str, optional, default='LogReg'
        Name of the classifier to use

    Returns
    -------
    y_pred : numpy.ndarray
        Array containing the predicted labels
    """
    clf = CLASSIFIERS[clf_name].fit(X_train, y_train)

    return clf.predict(X_test), clf.predict_proba(X_test)[:, 1]
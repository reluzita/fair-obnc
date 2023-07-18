from sklearn.linear_model import LogisticRegression
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import BaggingClassifier
import mlflow
import numpy as np

CLASSIFIERS = {
    'LogReg': LogisticRegression
}

class LabelCorrectionModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def correct(self, X:pd.DataFrame, y:pd.Series) -> pd.Series:
        """
        Corrects the labels of the given dataset

        Parameters
        ----------
        X : pd.DataFrame
            Dataset features
        y : pd.Series
            Labels to correct

        Returns
        -------
        y_corrected: pd.Series
            Corrected labels
        """
        pass

    @abstractmethod
    def log_params(self):
        pass

class OrderingBasedCorrection(LabelCorrectionModel):
    """
    Ordering-Based Correction algorithm

    Reference:
    Feng, Wei, and Samia Boukir. "Class noise removal and correction for image classification using ensemble margin." 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.

    Attributes
    ----------
    threshold : float
        Threshold for the margin of the ensemble classifier
    """
    def __init__(self, m):
        self.m = m

    def calculate_margins(self, X, y, bagging:BaggingClassifier):
        margins = pd.Series(dtype=float)
        for i in X.index:
            preds = [dt.predict(X.loc[i].values.reshape(1, -1))[0] for dt in bagging.estimators_]
            true_y = y.loc[i]

            v_1 = sum(preds)
            v_0 = len(preds) - v_1

            if true_y == 1:
                margins.loc[i] = (v_1 - v_0) / len(preds)
            else:
                margins.loc[i] = (v_0 - v_1) / len(preds)

        return margins

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)

        margins = self.calculate_margins(X.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)
        index = margins.index[:int(self.m*len(margins))]
        y_corrected.loc[index] = y_pred.loc[index]

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Ordering-Based Correction')

class FairOBNCRemoveSensitive(OrderingBasedCorrection):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC-rs)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    """
    def __init__(self, m, sensitive_attr):
        super().__init__('Fair-OBNC-rs', m)
        self.sensitive_attr = sensitive_attr

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()
        X_fair = X.drop(columns=self.sensitive_attr)

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X_fair, y)
        y_pred = pd.Series(bagging.predict(X_fair), index=y.index)

        margins = self.calculate_margins(X_fair.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)
        index = margins.index[:int(self.m*len(margins))]
        y_corrected.loc[index] = y_pred.loc[index]

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)

class FairOBNCOptimizeDemographicParity(OrderingBasedCorrection):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC-dp)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    prob : float
        Probability of correcting a label that does not contribute to balancing label distribution across sensitive groups
    """
    def __init__(self, m:float, sensitive_attr:str, prob:float):
        super().__init__('Fair-OBNC-dp', m)
        self.sensitive_attr = sensitive_attr
        self.prob = prob

    def dem_par_diff(self, X, y, attr):
        p_y1_g1 = len(y.loc[(X[attr] == 1) & (y == 1)]) / len(y.loc[X[attr] == 1])
        p_y1_g0 = len(y.loc[(X[attr] == 0) & (y == 1)]) / len(y.loc[X[attr] == 0])

        return p_y1_g1 - p_y1_g0

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)

        margins = self.calculate_margins(X.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        dem_par = self.dem_par_diff(X, y, self.sensitive_attr)
        n = int(self.m*len(margins))

        if dem_par == 0:
            y_corrected.loc[margins.index[:n]] = [(1 - y.loc[i]) for i in margins.index[:n]]

        else:
            corrected = 0
            for i in margins.index:
                if X.loc[i, self.sensitive_attr] == 0:
                    if y.loc[i] == int(dem_par < 0) or np.random.random() < self.prob:
                        y_corrected.loc[i] = y_pred.loc[i]
                        corrected += 1
                else:
                    if y.loc[i] == int(dem_par > 0) or np.random.random() < self.prob:
                        y_corrected.loc[i] =  y_pred.loc[i]
                        corrected += 1
                
                if corrected == n:
                    break

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)
        mlflow.log_param('prob', self.prob)

class FairOBNC(FairOBNCOptimizeDemographicParity):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    prob : float
        Probability of correcting a label that does not contribute to balancing label distribution across sensitive groups
    """
    def __init__(self, m:float, sensitive_attr:str, prob:float):
        super().__init__('Fair-OBNC', m, sensitive_attr, prob)

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()
        X_fair = X.drop(columns=self.sensitive_attr)

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X_fair, y)
        y_pred = pd.Series(bagging.predict(X_fair), index=y.index)

        margins = self.calculate_margins(X_fair.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        dem_par = self.dem_par_diff(X, y, self.sensitive_attr)
        n = int(self.m*len(margins))

        if dem_par == 0:
            y_corrected.loc[margins.index[:n]] = [(1 - y.loc[i]) for i in margins.index[:n]]

        else:
            corrected = 0
            for i in margins.index:
                if X.loc[i, self.sensitive_attr] == 0:
                    if y.loc[i] == int(dem_par < 0) or np.random.random() < self.prob:
                        y_corrected.loc[i] = y_pred.loc[i]
                        corrected += 1
                else:
                    if y.loc[i] == int(dem_par > 0) or np.random.random() < self.prob:
                        y_corrected.loc[i] =  y_pred.loc[i]
                        corrected += 1
                
                if corrected == n:
                    break

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)
        mlflow.log_param('prob', self.prob)


def get_label_correction_model(args) -> LabelCorrectionModel:
    """
    Initialize the label correction model

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing the correction algorithm and its parameters

    Returns
    -------
    model: LabelCorrectionModel
        Label correction model
    """

    if args.correction_alg == 'OBNC':
        return OrderingBasedCorrection(args.m)
    elif args.correction_alg == 'OBNC-remove-sensitive':
        return FairOBNCRemoveSensitive(args.m, args.sensitive_attr)
    elif args.correction_alg == 'OBNC-optimize-demographic-parity':
        return FairOBNCOptimizeDemographicParity(args.m, args.sensitive_attr, args.prob)
    elif args.correction_alg == 'OBNC-fair':
        return FairOBNC(args.m, args.sensitive_attr, args.prob)
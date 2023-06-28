from abc import ABC, abstractmethod
import pandas as pd
import pandas as pd
from sklearn.ensemble import BaggingClassifier
import numpy as np

class LabelCorrectionModel(ABC):
    def __init__(self, name):
        self.name = name

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

class OrderingBasedCorrection(LabelCorrectionModel):
    """
    Ordering-Based Correction algorithm

    Reference:
    Feng, Wei, and Samia Boukir. "Class noise removal and correction for image classification using ensemble margin." 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    """
    def __init__(self, m):
        super().__init__('OBNC')
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

class FairOBNCOptimizeDemographicParity(OrderingBasedCorrection):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC-dp)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    """
    def __init__(self, m:float, sensitive_attr:str):
        super().__init__('Fair-OBNC-dp', m)
        self.sensitive_attr = sensitive_attr

    def dem_par_diff(self, X, y, attr):
        p_y1_g1 = len(y.loc[(X[attr] == 1) & (y == 1)]) / len(y.loc[X[attr] == 1])
        p_y1_g0 = len(y.loc[(X[attr] == 0) & (y == 1)]) / len(y.loc[X[attr] == 0])

        return p_y1_g1 - p_y1_g0
    
    def correct_dem_par(self, X, y, y_pred, margins):
        y_corrected = y.copy()

        n = int(self.m*len(margins))
        corrected = 0

        for i in margins.index:
            dem_par = self.dem_par_diff(X, y, self.sensitive_attr)
            if dem_par == 0:
                y_corrected.loc[i] = y_pred.loc[i]
                corrected += 1
            else:
                if X.loc[i, self.sensitive_attr] == 0:
                    if y.loc[i] == int(dem_par < 0):
                        y_corrected.loc[i] = y_pred.loc[i]
                        corrected += 1
                else:
                    if y.loc[i] == int(dem_par > 0):
                        y_corrected.loc[i] =  y_pred.loc[i]
                        corrected += 1
            
            if corrected == n:
                break

        return y_corrected

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)

        margins = self.calculate_margins(X.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        y_corrected = self.correct_dem_par(X, y, y_pred, margins)

        return y_corrected

class FairOBNC(FairOBNCOptimizeDemographicParity):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    """

    def __init__(self, m:float, sensitive_attr:str):
        super().__init__('Fair-OBNC', m, sensitive_attr)

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()
        X_fair = X.drop(columns=self.sensitive_attr)

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X_fair, y)
        y_pred = pd.Series(bagging.predict(X_fair), index=y.index)

        margins = self.calculate_margins(X_fair.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        y_corrected = self.correct_dem_par(X_fair, y, y_pred, margins)

        return y_corrected
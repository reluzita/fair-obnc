import pandas as pd
import random
import os

def get_noisy_labels(noise_type, noise_rate, dataset, sensitive_attr, y, group, set):
    """
    Read saved noisy labels or generate noisy labels from clean labels

    Parameters
    ----------
    noise_type : str
        Type of noise to inject
    noise_rate : float
        Amount of noise to inject
    dataset : str
        Dataset name
    sensitive_attr : str
        Sensitive attribute name
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    set : str
        Dataset split

    Returns
    -------
    pd.Series
        Noisy labels
    """

    dir = f'data/{dataset}_{sensitive_attr}/{noise_type}/{noise_rate}'

    if os.path.exists(f'{dir}/{set}_labels.csv'):
        return pd.read_csv(f'{dir}/{set}_labels.csv', index_col=0)['y']

    y_noisy = inject_noise(y, group, noise_type, noise_rate)

    if not os.path.exists(f'data/{dataset}_{sensitive_attr}'):
        os.mkdir(f'data/{dataset}_{sensitive_attr}')

    if not os.path.exists(f'data/{dataset}_{sensitive_attr}/{noise_type}'):
        os.mkdir(f'data/{dataset}_{sensitive_attr}/{noise_type}')

    if not os.path.exists(dir):
        os.mkdir(dir)

    y_noisy.to_csv(f'{dir}/{set}_labels.csv', index=True)
    return y_noisy


def inject_noise(y:pd.Series, group:pd.Series, noise_type, noise_rate):
    """
    Inject noise into clean labels

    Parameters
    ----------
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    noise_type : str
        Type of noise to inject
    noise_rate : float
        Amount of noise to inject

    Returns
    -------
    pd.Series
        Noisy labels
    """
    if noise_type == 'random':
        return random_noise(y, noise_rate)
    elif noise_type == 'flip':
        return flip_noise(y, group, noise_rate)
    elif noise_type == 'bias':
        return bias_noise(y, group, noise_rate)
    elif noise_type == 'balanced_bias':
        return balanced_bias_noise(y, group, noise_rate)
    else:
        raise Exception('Invalid noise type')

def random_noise(y:pd.Series, noise_rate):
    random.seed(0)
    y_noisy = y.copy()

    change = random.sample(list(y.index), int(noise_rate * len(y)))
    for i in change:
        y_noisy.loc[i] = 1 - y.loc[i]

    return y_noisy

def flip_noise(y:pd.Series, group:pd.Series, noise_rate):
    random.seed(0)
    y_noisy = y.copy()

    for i in group.loc[group == 1].index:
        if random.random() < noise_rate:
            y_noisy.loc[i] = 1 - y.loc[i] 

    return y_noisy

def bias_noise(y:pd.Series, group:pd.Series, noise_rate):
    random.seed(0)
    y_noisy = y.copy()

    for i in group.loc[group == 1].index:
        if random.random() < noise_rate:
            y_noisy.loc[i] = 1

    return y_noisy

def balanced_bias_noise(y:pd.Series, group:pd.Series, noise_rate):
    random.seed(0)
    y_noisy = y.copy()

    for i in y_noisy.index:
        if random.random() < noise_rate:
            if group.loc[i] == 1:
                y_noisy.loc[i] = 1
            else:
                y_noisy.loc[i] = 0

    return y_noisy
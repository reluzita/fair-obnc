import argparse
import warnings
import logging
import random
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from label_correction import get_label_correction_model
from train import fit_predict
from evaluation import evaluate
from format_data import get_data
from sklearn.model_selection import train_test_split
import os
import pickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    warnings.filterwarnings("ignore")
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Systematic evaluation of label noise correction methods.')
    parser.add_argument('dataset', type=str, help='OpenML dataset id', choices=['adult', 'german', 'compas', 'ricci', 'diabetes', 'breastw', 'titanic'])
    parser.add_argument('sensitive_attr', type=str, help='Sensitive attribute')
    parser.add_argument('correction_alg', type=str, help='Label noise correction algorithm', choices=['OBNC', 'OBNC-remove-sensitive', 'OBNC-optimize-demographic-parity', 'OBNC-fair'])
    parser.add_argument('--test_size', type=float, help='Test set size', required=False, default=0.2)
    parser.add_argument('--model', type=str, help='Classification algorithm to use', required=False, default='LogReg', choices=['LogReg', 'DT'])
    parser.add_argument('--m', type=float, help='Percentage of labels to correct for Ordering-based correction', required=False, default=0.2)
    parser.add_argument('--prob', type=float, help='Probability for random correction for OBNC-optimize-demographic-parity', required=False, default=0)

    args = parser.parse_args()

    exp_name = f'{args.dataset}_{args.sensitive_attr}_{args.correction_alg}'
    mlflow.set_experiment(exp_name)
    print(f'Starting experiment: {exp_name}')

    run_tag = f'{datetime.now().strftime("%d%m%Y_%H%M%S")}'

    # get data
    X, y = get_data(args.dataset)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0, stratify=y)

    label_correction_model = get_label_correction_model(args)
    y_train_corrected = label_correction_model.correct(X_train, y_train)
    y_test_corrected = label_correction_model.correct(X_test, y_test)

    if not os.path.exists(f'correction/{args.dataset}_{args.sensitive_attr}'):
        os.makedirs(f'correction/{args.dataset}_{args.sensitive_attr}')
    
    y_train_corrected.to_csv(f'correction/{args.dataset}_{args.sensitive_attr}/{args.correction_alg}_train.csv', index=True)
    y_test_corrected.to_csv(f'correction/{args.dataset}_{args.sensitive_attr}/{args.correction_alg}_test.csv', index=True)
    
    for test_set in ['noisy', 'corrected']:
        for train_set in ['noisy', 'corrected']:
            with mlflow.start_run(tags={'train_set':train_set, 'test_set':test_set, 'run':run_tag}) as run:
                mlflow.log_param('dataset', args.dataset)
                mlflow.log_param('test_size', args.test_size)
                label_correction_model.log_params()
                mlflow.log_param('classifier', args.model)

                if train_set == 'noisy':
                    y_pred, y_pred_proba = fit_predict(X_train, y_train, X_test, args.model)
                else:
                    if y_train_corrected.unique().shape[0] == 1:
                        y_pred = y_test_corrected.copy()
                        y_pred_proba = y_test_corrected.copy()
                        print('After noise correction all labels are the same')
                    else:
                        y_pred, y_pred_proba = fit_predict(X_train, y_train_corrected, X_test, args.model)
                
                if not os.path.exists(f'predictions/{test_set}/{args.dataset}_{args.sensitive_attr}'):
                    os.makedirs(f'predictions/{test_set}/{args.dataset}_{args.sensitive_attr}')

                filename = 'noisy' if train_set == 'noisy' else args.correction_alg

                with open(f'predictions/{test_set}/{args.dataset}_{args.sensitive_attr}/{filename}.pkl', 'wb') as f:
                    pickle.dump(y_pred_proba, f)

                mlflow.log_param('senstive_attr', args.sensitive_attr)

                if test_set == 'noisy':
                    evaluate(y_test, y_pred_proba, X_test[args.sensitive_attr])
                else:
                    evaluate(y_test_corrected, y_pred_proba, X_test[args.sensitive_attr])


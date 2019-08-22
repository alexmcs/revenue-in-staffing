import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from collections import OrderedDict
import variable_selection
reload(variable_selection)

MAE = metrics.mean_absolute_error
median = metrics.median_absolute_error
MSE = metrics.mean_squared_error
expl_var = metrics.explained_variance_score
def RMSE(y_true, y_pred):
    return math.sqrt(MSE(y_true, y_pred))
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
R2 = metrics.r2_score
roc_auc = metrics.roc_auc_score
def pr_auc(y_true, y_pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    area = metrics.auc(recall, precision)
    return area
accuracy = metrics.accuracy_score
precision = metrics.precision_score
recall = metrics.recall_score
F1 = metrics.f1_score

def ninety_percentile(y_true, y_pred, n=90):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.percentile(np.abs(y_true - y_pred), n)


def mean_half_life_err(y_true, y_pred, half_life=30):
    over_err = np.where(np.array(y_true) < np.array(y_pred), np.array(y_pred) - np.array(y_true), 0)
    under_err = np.where(np.array(y_true) > np.array(y_pred), np.array(y_true) - np.array(y_pred), 0)

    half_life_err = (2 ** (-np.array(y_true) / half_life)) * (over_err + 2 * under_err)

    return np.mean(half_life_err)

def metrics_counting_clf_table(numeric_data_columns, X_train, X_test, y_train, y_test, y_pred):

    base_stratified = []
    base_major = []
    model_metrics = []

    ind = ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'F1']

    dummy_stratified_clf = DummyClassifier(strategy='stratified')
    dummy_major_clf = DummyClassifier(strategy='most_frequent')

    dummy_stratified_clf.fit(X_train[numeric_data_columns], y_train)
    dummy_major_clf.fit(X_train[numeric_data_columns], y_train)

    y_base_stratified = dummy_stratified_clf.predict_proba(X_test[numeric_data_columns])[:, 1]
    y_base_major = dummy_major_clf.predict_proba(X_test[numeric_data_columns])[:, 1]

    base_stratified.append(roc_auc(y_test, y_base_stratified))
    base_stratified.append(pr_auc(y_test, y_base_stratified))
    base_stratified.append(accuracy(y_test, y_base_stratified))
    base_stratified.append(precision(y_test, y_base_stratified))
    base_stratified.append(recall(y_test, y_base_stratified))
    base_stratified.append(F1(y_test, y_base_stratified))

    base_major.append(roc_auc(y_test, y_base_major))
    base_major.append(pr_auc(y_test, y_base_major))
    base_major.append(accuracy(y_test, y_base_major))
    base_major.append(precision(y_test, y_base_major))
    base_major.append(recall(y_test, y_base_major))
    base_major.append(F1(y_test, y_base_major))

    model_metrics.append(roc_auc(y_test, y_pred))
    model_metrics.append(pr_auc(y_test, y_pred))
    model_metrics.append(accuracy(y_test, np.round(y_pred)))
    model_metrics.append(precision(y_test, np.round(y_pred)))
    model_metrics.append(recall(y_test, np.round(y_pred)))
    model_metrics.append(F1(y_test, np.round(y_pred)))

    stratified_metrics = pd.DataFrame(base_stratified)
    major_metrics = pd.DataFrame(base_major)
    model_metrics = pd.DataFrame(model_metrics)

    result_metrics = pd.concat([stratified_metrics, major_metrics, model_metrics], axis=1)
    result_metrics.index = ind
    result_metrics.columns = ['Baseline_Stratified_Model:', 'Baseline_Major_Model:', 'Iteration_Model:']

    return result_metrics


def metrics_counting_reg(y_base, y_pred, y_test):
    # counting basic metrics for base-line prediction and model prediction

    base_metrics = []
    model_metrics = []

    ind = ['mean', 'MAE', 'median', '90-percentile', 'MSE', 'RMSE', 'MAPE', 'R2', 'explained variance']

    base_metrics.append(np.mean(y_test))
    base_metrics.append(MAE(y_test, y_base))
    base_metrics.append(median(y_test, y_base))
    #base_metrics.append(mean_half_life_err(y_test, y_base))
    base_metrics.append(ninety_percentile(y_test, y_base))
    base_metrics.append(MSE(y_test, y_base))
    base_metrics.append(RMSE(y_test, y_base))
    base_metrics.append(MAPE(y_test, y_base))
    base_metrics.append(R2(y_test, y_base))
    base_metrics.append(expl_var(y_test, y_base))
    model_metrics.append(np.mean(y_test))
    model_metrics.append(MAE(y_test, y_pred))
    model_metrics.append(median(y_test, y_pred))
    #model_metrics.append(mean_half_life_err(y_test, y_pred))
    model_metrics.append(ninety_percentile(y_test, y_pred))
    model_metrics.append(MSE(y_test, y_pred))
    model_metrics.append(RMSE(y_test, y_pred))
    model_metrics.append(MAPE(y_test, y_pred))
    model_metrics.append(R2(y_test, y_pred))
    model_metrics.append(expl_var(y_test, y_pred))

    base_metrics = pd.DataFrame(base_metrics)
    model_metrics = pd.DataFrame(model_metrics)

    metrics = pd.concat([base_metrics, model_metrics], axis=1)
    metrics.index = ind
    metrics.columns = ['Baseline_Model:', 'Iteration_Model:']
    metrics['Improvement:'] = 1 - metrics['Iteration_Model:'] / metrics['Baseline_Model:']

    return metrics


def mean_absolute_error(data, parameter = 'predictions'):
    df = data.copy()

    cases = [
        (
            parameter + ' 60+ days',
            (60 < df[parameter])
        ),
        (
            parameter + ' 30-60 days',
            (30 < df[parameter]) & (df[parameter] <= 60)
        ),
        (
            parameter + ' 0-30 days',
            (0 < df[parameter]) & (df[parameter] <= 30)
        )
    ]

    rows = [
        [
            ('case', desc),
            ('baseline MAE', '%.2f' % (abs(df[mask].target - df[mask].baseline)).mean()),
            ('model MAE', '%.2f' % (abs(df[mask].target - df[mask].predictions)).mean())
        ] for desc, mask in cases
    ]

    results = pd.DataFrame([OrderedDict(row) for row in rows])
    results['improvement'] = np.round(1 - results['model MAE'].astype(float) / results['baseline MAE'].astype(float), 2)

    return results
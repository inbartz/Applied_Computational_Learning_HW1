import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from numpy.core import mean, std
from scipy.sparse import isspmatrix_csc
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SVMSMOTE
import warnings
import lightgbm as lgb

# Variables
Not_blood_related = ['Patient ID', 'Patient age quantile',
                      'Patient addmited to regular ward (1=yes, 0=no)',
                      'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                      'Patient addmited to intensive care unit (1=yes, 0=no)',
                      'Respiratory Syncytial Virus',
                      'Influenza A', 'Influenza B', 'Parainfluenza 1',
                      'CoronavirusNL63', 'Rhinovirus/Enterovirus',
                      'Coronavirus HKU1', 'Parainfluenza 3', 'Strepto A',
                      'Chlamydophila pneumoniae', 'Adenovirus',
                      'Parainfluenza 4', 'Coronavirus229E',
                      'CoronavirusOC43', 'Inf A H1N1 2009',
                      'Bordetella pertussis', 'Metapneumovirus',
                      'Parainfluenza 2', 'Influenza B, rapid test',
                      'Influenza A, rapid test']
Required_features = ['Lactic Dehydrogenase', 'Aspartate transaminase', 'Alanine transaminase']
target_col_name = 'SARS-Cov-2 exam result'
random_seed = 42
k_neighbors = 5
# RF params
n_estimators = [10, 20, 30] + [i for i in range(45, 105, 5)]
max_depth = [2, 4, 8, 16, 32, 64]
# XGBoost params
learning_rate = [0.1, 0.05, 0.01]



def read_data(path):
    data = pd.read_csv(path + "dataset.csv")
    return data


def correlation(feature_name1, feature_name2, feature_array1, feature_array2):
    arr1 = np.zeros(len(feature_array1))
    arr2 = np.zeros(len(feature_array2))
    for index in range(len(feature_array1)):
        if feature_array1[index] is not None and feature_array2[index] is not None:
            arr1[index] = feature_array1[index]
            arr2[index] = feature_array2[index]
    if len(arr1) == len(arr2):
        print(np.corrcoef(arr1, arr2))
        plt.scatter(arr1, arr2)
        plt.title('A plot to show the correlation between {0} and {1}'.format(feature_name1, feature_name2))
        plt.xlabel(feature_name1)
        plt.ylabel(feature_name2)
        plt.plot(np.unique(arr1), np.poly1d(np.polyfit(arr1, arr2, 1))(np.unique(arr1)), color='yellow')
        plt.show()


def check_for_correlations(data):
    data = pd.DataFrame(fill_none_values(data), columns=data.columns)
    for index_first in range(data.shape[1]-1):
        for index_second in range(index_first+1, data.shape[1]):
            print("1: "+str(data.columns[index_first])+" 2: "+str(data.columns[index_second]))
            correlation(data.columns[index_first], data.columns[index_second],
                        data.iloc[: , index_first].values, data.iloc[: , index_second].values)


def choose_features(data, features_to_drop, Required_features):
    # Droop all the features that are not related to blood
    data.drop(features_to_drop, axis='columns', inplace=True)
    # Set the threshold value
    threshold_min_values_count = int(0.05 * len(data))
    # This loop remove all features that the number of non nane values in them is 5% from all rows.
    for col in data.columns:
        if threshold_min_values_count > data[col].count() and col not in Required_features:
            data.drop(col, axis='columns', inplace=True)
            # Clean all row that have more then 2 features with None values.
    data.dropna(thresh=2, inplace=True)
    return data

def get_target_values(data , target_col_name):
    target_values = data[target_col_name]
    target_values = target_values.replace({'negative': 0, 'positive': 1})
    return np.array(target_values.values.tolist())

# preform pre-process includes:
    #   remove features that are not related to blood
    #   remove feature with more then 95% none values
    #   remove rows with more then 2
    #   Change the target value to binary
def data_pre_process(data, features_to_drop, Required_features, target_col_name):
    clean_data = choose_features(data, features_to_drop, Required_features)
    target_values = get_target_values(clean_data, target_col_name)
    clean_data.drop(target_col_name, axis='columns', inplace=True)
    return clean_data, target_values

def fill_none_values(data):
    new_data = pd.DataFrame(data)
    imputer = IterativeImputer(random_state=0)
    imputer.fit(new_data)
    data_without_null = imputer.transform(new_data)
    return data_without_null

def fill_nones(X_train, X_test):
    imputer = IterativeImputer(random_state=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def train_and_evaluate(X, Y, model, grid_variables):
    f1_max_score = 0
    final_model = None
    svm_smote = SVMSMOTE(random_state=12, k_neighbors=k_neighbors)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    # X = fill_none_values(X)
    # X , Y = svm_smote.fit_resample(X, Y)
    # print(type(X))
    # print(type(Y))
    X = X.to_numpy()
    print(type(X))
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        Y_train, Y_test = Y[train_index], Y[test_index]
        X_train, X_test = fill_nones(X_train, X_test)
        print(type(X_train))
        print(type(Y_train))
        X_train, Y_train = svm_smote.fit_resample(X_train, Y_train)
        X_test, Y_test = svm_smote.fit_resample(X_test, Y_test)
        inner_cv = KFold(shuffle=True, random_state=random_seed)
        grid = GridSearchCV(model, grid_variables, scoring='f1', cv=inner_cv, refit=True)
        grid_results = grid.fit(X_train, Y_train)
        model = grid_results.best_estimator_
        yhat = model.predict(X_test)
        f1_model_score = f1_score(Y_test, yhat)
        if f1_model_score > f1_max_score:
            f1_max_score = f1_model_score
            final_model = model
        print("final model is:"+ str(final_model))
        print("current model is:"+ str(model))
        print(f1_model_score)
    return final_model

def evaluated_by_confusion_matrix(Y_test, yhat):
    matrix = confusion_matrix(Y_test, yhat)
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]
    return tp / float(tp + fn), tn / float(tn + fp)

def show_model_evaluation(list, model_name):
    print(f'==== {model_name} ===')
    print('Accuracy: %.3f (%.3f)' % (mean(list[0]), std(list[0])))
    print('F1: %.3f (%.3f)' % (mean(list[1]), std(list[1])))
    print('Sensitivity: %.3f (%.3f)' % (mean(list[2]), std(list[2])))
    print('Specificity: %.3f (%.3f)' % (mean(list[3]), std(list[3])))
    print('AUROC: %.3f (%.3f)' % (mean(list[4]), std(list[4])))

def run_and_train_model(X, Y, model, num_of_runs):
    # evaluation parameters
    accuracy_list = []
    f1_scores_list = []
    sensitivity_list = []
    specificity_list = []
    ROC_area_list = []
    final_model = None
    f1_max_score = 0
    random_states = list(range(10, 101, 10))
    for index in range(num_of_runs):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_states[index])
        X_train, X_test = fill_nones(X_train, X_test)
        svm_smote = SVMSMOTE(random_state=12, k_neighbors=k_neighbors)
        X_train, Y_train = svm_smote.fit_resample(X_train, Y_train)
        model.fit(X_train, Y_train)
        yhat = model.predict(X_test)
        # evaloation values
        accuracy = accuracy_score(Y_test, yhat)
        accuracy_list.append(accuracy)
        f1_model_score = f1_score(Y_test, yhat)
        f1_scores_list.append(f1_model_score)
        sensitivity, specificity = evaluated_by_confusion_matrix(Y_test, yhat)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        ROC_area = roc_auc_score(Y_test, yhat)
        ROC_area_list.append(ROC_area)
        # print("accuracy - " + str(accuracy))
        # print("f1 - " + str(f1_model_score))
        # print("conf_sensitivity - " + str(sensitivity))
        # print("conf_specificity - " + str(specificity))
        # print("auroc - " + str(ROC_area))
        if f1_model_score > f1_max_score:
            f1_max_score = f1_model_score
            final_model = model
    eval_params = [accuracy_list, f1_scores_list, sensitivity_list, specificity_list, ROC_area_list]
    return final_model, eval_params

def logistic_regression_model(X, Y):
    model = LogisticRegression(random_state=random_seed)
    grid = {}
    best_model = train_and_evaluate(X, Y, model, grid)
    best_model, evaluation_params = run_and_train_model(X, Y, best_model, 10)
    show_model_evaluation(evaluation_params, "LogisticRegression")
    return best_model

def random_forest_best_model(X, Y):
    # Random Forest
    model = RandomForestClassifier(random_state=random_seed)
    grid = {'n_estimators': n_estimators,
            'max_depth': max_depth}
    best_model = train_and_evaluate(X, Y, model, grid)
    best_model, evaluation_params = run_and_train_model(X, Y, best_model, 10)
    show_model_evaluation(evaluation_params, "RF")
    return best_model

def XGBoost_model(X, Y):
    model = xgb.XGBClassifier(random_state=random_seed)
    grid = {'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate}
    best_model = train_and_evaluate(X, Y, model, grid)
    best_model, evaluation_params = run_and_train_model(X, Y, best_model, 10)
    show_model_evaluation(evaluation_params, "XGBoost")
    return best_model

def CatBoost_model(X, Y):
    model = CatBoostClassifier(iterations=5, learning_rate=0.1)
    best_model, evaluation_params = run_and_train_model(X, Y, model, 10)
    show_model_evaluation(evaluation_params, "CatBoost")
    return best_model

def LightGBM(X, Y):
    model = lgb.LGBMClassifier()
    best_model, evaluation_params = run_and_train_model(X, Y, model, 10)
    show_model_evaluation(evaluation_params, "LightGBM")
    return best_model


if __name__ == '__main__':
    path = sys.argv[1]
    # read data from csv file
    all_data = read_data(path)
    X, Y = data_pre_process(all_data, Not_blood_related, Required_features, target_col_name)
    # check_for_correlations(X)
    lr_model = logistic_regression_model(X, Y)
    rf_model = random_forest_best_model(X, Y)
    XGBoost_best = XGBoost_model(X, Y)
    # create new data with 5 new features

    cb_model = CatBoost_model(X, Y)
    lGBM_model = LightGBM(X, Y)
    #print("the best model = " + str(lr_model))



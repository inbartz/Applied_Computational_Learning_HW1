import sys

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

# Variables
Not_blood_related = ['ï»¿Patient ID', 'Patient age quantile',
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
# path = "C:\\Users\\inbar\\Desktop\\LearningHW1\\"
Required_features = ['Lactic Dehydrogenase', 'Aspartate transaminase', 'Alanine transaminase']
target_col_name = 'SARS-Cov-2 exam result'


def read_data(path):
    data = pd.read_csv(path + "dataset.csv", encoding='latin-1')
    return data


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
    return target_values.to_frame()


def data_pre_process(data, features_to_drop, Required_features, target_col_name):
    clean_data = choose_features(data, features_to_drop, Required_features)
    target_values = get_target_values(clean_data, target_col_name)
    clean_data.drop(target_col_name, axis='columns', inplace=True)
    return clean_data, target_values


def fill_none_values(data, initial_strategy):
    new_data = pd.DataFrame(data)
    #imputer = IterativeImputer(random_state=0,  initial_strategy=initial_strategy)
    imputer = IterativeImputer(random_state=0)
    imputer.fit(new_data)
    data_without_null = imputer.transform(new_data)
    df_data = pd.DataFrame(data_without_null, columns=data.columns)
    return df_data


def random_forest_classification(X, y):
    model = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)


if __name__ == '__main__':
    path = sys.argv[1]
    # read data from csv file
    all_data = read_data(path)
    # preform pre-process includes:
    #   remove features that are not related to blood
    #   remove feature with more then 95% none values
    #   remove rows with more then 2
    #   Change the target value to binary
    X, y = data_pre_process(all_data, Not_blood_related, Required_features, target_col_name)
    print("This is X - features and values: "+str(X))
    print("This is y - target values 0=negative, 1=positive: " + str(y))
    # Need to check which strategy gives better result
    # X_mean_insteadof_none = fill_none_values(X, 'mean')
    # X_median_insteadof_none = fill_none_values(X, 'median')
    # print("This is X with mean values - features and values: " + str(X_mean_insteadof_none))
    # print("This is X with median values - features and values: " + str(X_median_insteadof_none))
    X = fill_none_values(X, '')
    print("This is X without none values - features and values: "+str(X))
    random_forest_classification(X, y)


import pandas as pd

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
path = "C:\\Users\\inbar\\Desktop\\LearningHW1\\"
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


if __name__ == '__main__':
    # read data from csv file
    all_data = read_data(path)
    # preform pre-process includes:
    # remove features that are not related to blood
    # remove feature with more then 95% none values
    # remove rows with more then 2
    # Change the target value to binary
    X, Y = data_pre_process(all_data, Not_blood_related, Required_features, target_col_name)
    print("This is X - features and values: "+str(X))
    print("This is Y - target values 0=negative, 1=positive: " + str(Y))
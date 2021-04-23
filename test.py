import numpy as np
import pandas as pd
from numpy.core import mean, std
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split, cross_val_score
from imblearn.over_sampling import SVMSMOTE, SMOTE
import shap
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

ADD_FEATURES=False


def load_data():
    """
    Loading data based on the fact that it is in the same folder as the script
    :return: dataframe
    """
    data=pd.read_csv('C:\\Users\\inbar\\Desktop\\LearningHW1\\dataset.csv')
    return data


def data_preparation(data):
    """
    Cleaning the data as described in Section 5.2. Leaving only the significant features as they were defined,
    then removing all the entries that are "NaN" in all the columns, and renaming the columns to their acronyms
    as done in the article.
    :param data: Dataframe
    :return: X and y with 608 observations
    """
    data_col=data.columns
    blood_col=['Alanine transaminase', 'Aspartate transaminase', 'Basophils', 'Creatinine', 'Eosinophils',
               'Hematocrit', 'Hemoglobin', 'Lactic Dehydrogenase', 'Lymphocytes',
               'Mean corpuscular hemoglobin concentration (MCHC)', 'Mean corpuscular volume (MCV)',
               'Mean platelet volume ', 'Monocytes', 'Neutrophils', 'Platelets', 'Potassium',
               'Proteina C reativa mg/dL', 'Red blood cell distribution width (RDW)', 'Red blood Cells', 'Sodium',
               'Urea', 'Leukocytes', 'Mean corpuscular hemoglobin (MCH)', 'SARS-Cov-2 exam result']
    non_blood_col=['Adenovirus',
                   'Bordetella pertussis',
                   'Chlamydophila pneumoniae',
                   'Coronavirus HKU1',
                   'Coronavirus229E',
                   'CoronavirusNL63',
                   'CoronavirusOC43',
                   'Inf A H1N1 2009',
                   'Influenza A',
                   'Influenza A, rapid test',
                   'Influenza B',
                   'Influenza B, rapid test',
                   'Metapneumovirus',
                   'Parainfluenza 1',
                   'Parainfluenza 2',
                   'Parainfluenza 3',
                   'Parainfluenza 4',
                   'Patient addmited to intensive care unit (1=yes, 0=no)',
                   'Patient addmited to regular ward (1=yes, 0=no)',
                   'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                   'Patient age quantile',
                   'Patient ID',
                   'Respiratory Syncytial Virus',
                   'Rhinovirus/Enterovirus',
                   'Strepto A']
    for col in data_col:
        prec_nan=data[col].isna().sum() / len(data[col])
        if col not in blood_col and prec_nan > 0.95:
            data=data.drop(columns=[col])
    data=data.drop(columns=non_blood_col)
    data=data.rename(columns={'Alanine transaminase': 'ALT',
                              'Aspartate transaminase': 'AST',
                              'Basophils': 'BAY',
                              'Creatinine': 'CREAT',
                              'Eosinophils': 'EOS',
                              'Hematocrit': 'HCT',
                              'Hemoglobin': 'HGB',
                              'Lactic Dehydrogenase': 'LDH',
                              'Leukocytes': 'WBC',
                              'Lymphocytes': 'LYM',
                              'Mean corpuscular hemoglobin (MCH)': 'MCH',
                              'Mean corpuscular hemoglobin concentration (MCHC)': 'MCHC',
                              'Mean corpuscular volume (MCV)': 'NCV',
                              'Mean platelet volume ': 'MPV',
                              'Monocytes': 'MONO',
                              'Neutrophils': 'NEU',
                              'Platelets': 'PLT',
                              'Potassium': 'K',
                              'Proteina C reativa mg/dL': 'CRP',
                              'Red blood cell distribution width (RDW)': 'RWD',
                              'Red blood Cells': 'RBC',
                              'SARS-Cov-2 exam result': 'result',
                              'Sodium': 'Na',
                              'Urea': 'UREA'})
    X=data.iloc[:, 1:]
    y=data.iloc[:, 0]
    X=X.dropna(how='all')
    y=y[X.index].to_frame()
    y=y.replace({'negative': 0, 'positive': 1})
    return X, y


def impute(data):
    """
    Using “IterativeImputer” technique from Scikit-learn package to seal with the null values
    :param data: dataframe, X, with missing values
    :return: Imputed data
    """
    imp_mean=IterativeImputer(random_state=0)
    imp_mean.fit(data)
    data=imp_mean.transform(data)
    return data


def train_the_model(X, y, model, model_name):
    """
    Retrain the given model with the selected hyperparameters, in 10 iterations where for each iteration
    splitting the data randomly to 80-20 for train-test, as described in section 5.3.
    The following evaluation metrics are calculated: accuracy, f1–score, sensitivity and specificity.
    :param X: numpy ndarray, X values
    :param y: numpy ndarray, target
    :param model: the best predictive model with the chosen hyperparameters, estimator object
    :param model_name: model's name
    :return: The model that achieved the best F1-score ih the 10 iteration process
    """
    random_states=list(range(10, 101, 10))
    f1_scores, acc_scores, sen_scores, spe_scores, auroc_scores, best_f1, best_model=[], [], [], [], [], 0, None
    for i in range(10):
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=random_states[i])
        sm=SVMSMOTE(random_state=12)
        X_train, y_train=sm.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        yhat=model.predict(X_test)
        # Calculate metrics
        f1=f1_score(y_test, yhat, average='macro')
        acc=accuracy_score(y_test, yhat)
        auroc=roc_auc_score(y_test, yhat)
        # Creating the confusion matrix
        conf_matrix=confusion_matrix(y_test, yhat)
        TP=conf_matrix[1][1]
        TN=conf_matrix[0][0]
        FP=conf_matrix[0][1]
        FN=conf_matrix[1][0]
        # calculate the sensitivity
        conf_sensitivity=(TP / float(TP + FN))
        # calculate the specificity
        conf_specificity=(TN / float(TN + FP))
        sen_scores.append(conf_sensitivity)
        spe_scores.append(conf_specificity)
        auroc_scores.append(auroc)
        f1_scores.append(f1)
        acc_scores.append(acc)
        print("acc - " + str(acc))
        print("f1 - " + str(f1))
        print("conf_sensitivity - " + str(conf_sensitivity))
        print("conf_specificity - " + str(conf_specificity))
        print("auroc - " + str(auroc))
        if f1 > best_f1:
            best_f1=f1
            best_model=model
    print(f'==== {model_name} ===')
    print('F1: %.3f (%.3f)' % (mean(f1_scores), std(f1_scores)))
    print('Accuracy: %.3f (%.3f)' % (mean(acc_scores), std(acc_scores)))
    print('Sensitivity: %.3f (%.3f)' % (mean(sen_scores), std(sen_scores)))
    print('Specificity: %.3f (%.3f)' % (mean(spe_scores), std(spe_scores)))
    print('AUROC: %.3f (%.3f)' % (mean(auroc_scores), std(auroc_scores)))
    return best_model


def run_hyper_parameter_optimization(X, y, model, random_grid, model_name):
    """
    Perform hyper-parameter optimization using nested cross-validation procedure as described in section 5.3.
    :param X: numpy ndarray, X values
    :param y: numpy ndarray, target
    :param model: predictive model, estimator object
    :param random_grid: dictionary with hyper-parameter to optimize
    :param model_name: name of the predictive model
    :return: Estimator that was chosen by the search, i.e. estimator which gave highest f1-score
    """
    best_f1, best_best_model, best_result=0, None, None
    cv_outer=KFold(n_splits=5, shuffle=True, random_state=42)
    sm=SMOTE(random_state=12, k_neighbors=5)
    X, y=sm.fit_resample(X, y)
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test=X[train_ix, :], X[test_ix, :]
        y_train, y_test=y[train_ix], y[test_ix]
        cv_inner=KFold(n_splits=5, shuffle=True, random_state=11)
        search=GridSearchCV(model, random_grid, scoring='f1', cv=cv_inner, refit=True)
        result=search.fit(X_train, y_train)
        best_model=result.best_estimator_
        yhat=best_model.predict(X_test)
        f1=f1_score(y_test, yhat, average='macro')
        if f1 > best_f1:
            best_f1=f1
            best_best_model=best_model
            best_result=result
        print("final model is:" + str(best_best_model))
        print("current model is:" + str(model))
        print(best_f1)
    print(f"{model_name}: {best_result.best_params_}")
    return best_best_model


def run_shap_two_dimension(model, X_df):
    """
    Use for the Random Forest and LightGBM models
    Creating beeswarm plot from SHAP model, that display an information-dense summary of how the top features in a dataset impact the model’s output
    :param model: estimator object
    :param X_df: datframe without the target
    """
    explainer=shap.Explainer(model)
    shap_values=explainer(X_df)
    shap_values.values=shap_values.values[:, :, 1]
    shap_values.base_values=shap_values.base_values[:, 1]
    shap.plots.beeswarm(shap_values, max_display=50)
    shap.plots.bar(shap_values)


def run_shap_one_dimension(model, X_df):
    """
    Use for the XGBoost and Catboost models
    Creating beeswarm plot from SHAP model, that display an information-dense summary of how the top features in a dataset impact the model’s output
    :param model: estimator object
    :param X_df: datframe without the target
    """
    explainer=shap.Explainer(model)
    shap_values=explainer(X_df)
    shap.plots.beeswarm(shap_values, max_display=50)
    shap.plots.bar(shap_values)


def run_shap_lr(model, X_df):
    """
    Use for the Logistic Regression models
    Creating beeswarm plot from SHAP model, that display an information-dense summary of how the top features in a dataset impact the model’s output
    :param model: estimator object
    :param X_df: dataframe without the target
    """
    explainer=shap.LinearExplainer(model, X_df)
    shap_values=explainer(X_df)
    shap.plots.beeswarm(shap_values, max_display=50)
    shap.plots.bar(shap_values)


def add_features(X_df):
    """
    Creating new features from the raw features
    :param X_df: dataframe with the raw features
    :return: dataframe with the new features
    """
    X_df['WBC*EOS']=X_df['WBC'] * X_df['EOS']
    X_df['WBC*EOS']=(X_df['WBC*EOS'] - X_df['WBC*EOS'].min()) / (X_df['WBC*EOS'].max() - X_df['WBC*EOS'].min())
    X_df['PLT*EOS']=X_df['PLT'] * X_df['EOS']
    X_df['PLT*EOS']=(X_df['PLT*EOS'] - X_df['PLT*EOS'].min()) / (X_df['PLT*EOS'].max() - X_df['PLT*EOS'].min())
    X_df['PLT*WBC']=X_df['PLT'] * X_df['WBC']
    X_df['PLT*WBC']=(X_df['PLT*WBC'] - X_df['PLT*WBC'].min()) / (X_df['PLT*WBC'].max() - X_df['PLT*WBC'].min())
    X_df['WBC*BAY']=X_df['WBC'] * X_df['BAY']
    X_df['WBC*BAY']=(X_df['WBC*BAY'] - X_df['WBC*BAY'].min()) / (X_df['WBC*BAY'].max() - X_df['WBC*BAY'].min())
    X_df['MPV^2']=X_df['MPV'] * X_df['MPV']
    X_df['MPV^2']=(X_df['MPV^2'] - X_df['MPV^2'].min()) / (X_df['MPV^2'].max() - X_df['MPV^2'].min())
    X=X_df.to_numpy()
    return X, X_df


if __name__ == '__main__':
    data_df=load_data()
    print(data_df)
    X, y=data_preparation(data_df)
    print("This is X: " + str(X))
    print("This is y: " + str(y))
    columns=X.columns
    X=impute(X)
    X_df=pd.DataFrame(X, columns=columns)
    if ADD_FEATURES:
        X, X_df=add_features(X_df)
    y=np.array(y.values.tolist())
    # Grid Settings
    n_estimators=[10, 20, 30] + [i for i in range(45, 105, 5)]
    max_depth=[2, 4, 8, 16, 32, 64]
    learning_rate=[0.1, 0.05, 0.01]
    # Random Forest
    # rf_model=RandomForestClassifier(random_state=1)
    # rf_random_grid={'n_estimators': n_estimators,
    #                 'max_depth': max_depth}
    # rf_best_model=run_hyper_parameter_optimization(X, y, rf_model, rf_random_grid, 'rf')
    # rf_best_model=train_the_model(X, y, rf_best_model, 'rf')
    # run_shap_two_dimension(rf_best_model, X_df)
    # # XGBoost
    # xg_model=xgb.XGBClassifier(random_state=2)
    # xg_random_grid={'n_estimators': n_estimators,
    #                 'learning_rate': learning_rate,
    #                 'max_depth': max_depth}
    # xg_best_model=run_hyper_parameter_optimization(X, y, xg_model, xg_random_grid, 'xg')
    # xg_best_model=train_the_model(X, y, xg_best_model, 'xg')
    # run_shap_one_dimension(xg_best_model, X_df)
    # # Logistic Regression
    # lr_model=LogisticRegression(random_state=42)
    # lr_random_grid={}
    # lr_best_model=run_hyper_parameter_optimization(X, y, lr_model, lr_random_grid, 'lr')
    # lr_best_model=train_the_model(X, y, lr_best_model, 'lr')
    # run_shap_lr(lr_best_model, X_df)
    # # CatBoost
    # cat_model=CatBoostClassifier(iterations=5, learning_rate=0.1)
    # cat_model=train_the_model(X, y, cat_model, 'cat')
    # run_shap_one_dimension(cat_model, X_df)
    # LightGBM
    lgb_model=lgb.LGBMClassifier()
    lgb_model=train_the_model(X, y, lgb_model, 'lgb')
    run_shap_two_dimension(lgb_model, X_df)

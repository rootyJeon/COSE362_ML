import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score, cross_validate, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.feature_selection import RFE







def get_dataset_seperated(train_size = 0.8, validate_size = 0.0, cols_to_drop = ['track', 'artist', 'uri']):
    total_index = ["60", "70", "80", "90", "00", "10"]
    total_index_full = ["1960", "1970", "1980", "1990", "2000", "2010"]

    temp_ds_train = []
    temp_ds_test = []
    temp_ds_validate = []

    for i in range(len(total_index)):
        # 분리된 csv 병합
        dataset = pd.read_csv('data/dataset-of-{0}s.csv'.format(total_index[i]), header=0, delimiter=',')

        dataset_train = dataset.sample(frac=train_size, random_state=1, replace=True)
        dataset_test = dataset.drop(dataset_train.index)

        dataset_validate_sample = dataset_test.sample(frac=validate_size, random_state=1)
        dataset_validate = dataset_test.drop(dataset_validate_sample.index)

        dataset_train['decade'] = total_index_full[i]
        dataset_test['decade'] = total_index_full[i]
        dataset_validate['decade'] = total_index_full[i]

        temp_ds_train.append(dataset_train)
        temp_ds_test.append(dataset_test)
        temp_ds_validate.append(dataset_validate)

    dataset_train = pd.concat(temp_ds_train, axis=0, ignore_index=True)
    dataset_test = pd.concat(temp_ds_test, axis=0, ignore_index=True)
    dataset_validate = pd.concat(temp_ds_validate, axis=0, ignore_index=True)

    dataset_train.drop(cols_to_drop, axis=1, inplace=True)
    dataset_test.drop(cols_to_drop, axis=1, inplace=True)
    dataset_validate.drop(cols_to_drop, axis=1, inplace=True)

    dataset_train.to_csv('data/dataset_train_joined_flushed.csv')
    dataset_test.to_csv('data/dataset_test_joined_flushed.csv')
    dataset_validate.to_csv('data/dataset_validate_joined_flushed.csv')

    return dataset_train, dataset_test, dataset_validate


def generate_dataset():
    drops = ['track', 'artist', 'uri'] # 삭제할 attribute
    dataset_train, dataset_test, dataset_validate = get_dataset_seperated(train_size=0.8, validate_size=0.0, cols_to_drop=drops)

    return dataset_train, dataset_test, dataset_validate


def get_train_test(dataset_train=None, dataset_test=None, dataset_validate=None, dataset_full=None):
    if dataset_full is not None:
        x = dataset_full.drop(['target'], 1).values # 값을 numpy array로 바꾸기
        y = dataset_full['target'].values.ravel()

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        return x_train, x_test, y_train, y_test

    x_train = dataset_train.drop(['target'], 1).values
    y_train = dataset_train['target'].values.ravel()

    x_test = dataset_test.drop(['target'], 1).values
    y_test = dataset_test['target'].values.ravel()

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, x_test, y_test, model_name):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # 과정 1에 대한 각 모델 생성
    model = RandomForestClassifier()
    if model_name == "LogisticReg":
        model = LogisticRegression(random_state=42)

    elif model_name == "DT":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "MLP":
        model = MLPClassifier(max_iter=2000, random_state=42)

    elif model_name == "NB":
        model = GaussianNB()

    elif model_name == "RF":
        model = RandomForestClassifier(random_state=42)

    elif model_name == "SVM":
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        model = LinearSVC(random_state=42)

    else:
        print("Wrong Model!")

    # model fit
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Estimator: ", model)

    return model, y_pred


def train_rfe_model(x_train, y_train, x_test, y_test, model_name):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # 각 모델에 대한 feature selection 적용 모델 생성 (과정2)
    model = RandomForestClassifier()
    if model_name == "LogisticReg":
        model = RFE(LogisticRegression(random_state=42), n_features_to_select=10)

    elif model_name == "DT":
        model = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=10)

    elif model_name == "NB":
        model = GaussianNB()

    elif model_name == "RF":
        model = RFE(RandomForestClassifier(random_state=42), n_features_to_select=10)

    elif model_name == "SVM":
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        model = RFE(LinearSVC(random_state=42), n_features_to_select=10)

    else:
        print("Wrong Model!")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Estimator: ", model)

    return model, y_pred


def validate_model(x_train, y_train, model, cls=None):
    scores = cross_val_score(model, x_train, y_train, cv=5)
    # validation
    print("Validation Score: ", scores)
    print("Validation Avg Score: ", np.mean(scores))

    return np.mean(scores)

def evaluate_model(x_train, x_test, y_train, y_test, y_pred, model, cls=None):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score


    # 각 score 계산
    p_score = precision_score(y_test, y_pred)
    r_score = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cfmatrix = confusion_matrix(y_test, y_pred, labels=cls)
    accuracy_score_ = accuracy_score(y_test, y_pred)

    # 출력
    print("Accuracy(Test):", accuracy_score_ * 100)
    print("Precision:", p_score * 100)
    print("Recall:", r_score * 100)
    print("F1 score:", f1)
    print('Training MSE: ', np.mean((model.predict(x_train) - y_train) ** 2))
    print('Test model MSE', np.mean((model.predict(x_test) - y_test) ** 2))

    # confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cfmatrix, display_labels=cls)
    disp.plot()
    plt.show()

    return accuracy_score_, p_score, r_score, f1


def hyperparameter_search(x_train, y_train, x_test, y_test, model_name, params=None):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)



    # BayesSearchCV를 통해 Cross-Validation
    opt = ""

    if model_name == "LogisticReg":
        opt = BayesSearchCV(
            LogisticRegression(),
            {
                'C': (1e-4, 2e+0),
                'tol': (1e-9, 1e-1)
            },
            n_iter=5,
            cv=5,
            #verbose=2,
            random_state=0
        )


    elif model_name == "DT":
        opt = BayesSearchCV(
            DecisionTreeClassifier(),
            {
                'max_depth': Integer(1, 20),
                'max_features': Categorical(['auto', 'sqrt', 'log2']),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 20),
                'min_impurity_decrease': (0.0, 1.0),
                'criterion': Categorical(['gini', 'entropy']),
            },
            n_iter=10,
            cv=5,
            #verbose=2,
            random_state=0
        )

    elif model_name == "NB":
        opt = BayesSearchCV(
            GaussianNB(),
            {
                "var_smoothing": (5e-10, 15e-10)
            },
            n_iter=10,
            cv=5,
            #verbose=2,
        )

    elif model_name == "RF":
        opt = BayesSearchCV(
            RandomForestClassifier(),
            {
                'n_estimators': Integer(1, 500),
                'max_depth': Integer(1, 20),
                'max_features': Categorical(['auto', 'sqrt', 'log2']),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 20),
                'bootstrap': Categorical([True, False]),
                'criterion': Categorical(['gini', 'entropy'])
            },
            n_iter=5,
            cv=5,
            #verbose=2,
            random_state=0
        )

    elif model_name == "SVM":
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        opt = BayesSearchCV(
            LinearSVC(),
            {
                'C': (1e-2, 10),
                'tol': (1e-9, 1e-1)
            },
            n_iter=32,
            cv=5,
            #verbose=2,
            random_state=0
        )



    _ = opt.fit(x_train, y_train)
    '''
    print(opt.score(x_test, y_test))
    print(opt.best_params_)
    print(opt.best_score_)
    print(opt.best_estimator_)
    '''
    print("Estimator: ", opt.best_estimator_)
    print("Validation Avg Score of Best Model : ", opt.best_score_)
    return opt.best_score_, opt.best_params_, opt.best_estimator_




dataset_train, dataset_test, dataset_validate = generate_dataset()


x_train, x_test, y_train, y_test = get_train_test(dataset_train, dataset_test, dataset_validate)

model_types = ["SVM", "DT", "LogisticReg", "NB", "RF"]
goodmodels = []
goodscores = []

# 5개의 모델에 대해 보고서 과정 1~3 모델 생성 및 Cross-Validation 적용
for model_type in model_types:
    print("< Analysis of Model Using Default Parameters >\n")
    model, y_pred = train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_name=model_type)
    score1 = validate_model(x_train=x_train, y_train=y_train, model=model)

    print("\n=========================================\n")
    print("< Analysis of Model Using Feature Selection >\n")
    model_rfe, y_pred_rfe = train_rfe_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_name=model_type)
    score2= validate_model(x_train=x_train, y_train=y_train, model=model_rfe)

    print("\n=========================================\n")
    print("< Analysis of Model After Parameter Search >\n")
    score3, params, parambest_model = hyperparameter_search(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_name=model_type)

    print("=========================================")

    models = [model, model_rfe, parambest_model]
    scores = [score1, score2, score3]
    goodmodel = models[scores.index(max(scores))]
    print("Best Model using(" + model_type + "): ", goodmodel)
    print("=========================================\n\n")

    # 과정 1~3 중 높은 스코어를 기록한 모델이 goodscores에 append되고 goodscores에 있는 모델들 중에서 성능이 가장 좋은 최종 모델 선택
    goodmodels.append(goodmodel)
    goodscores.append(max(scores))

idx = goodscores.index(max(goodscores))

### bestmodel이 최종적으로 선택된 분류기 ###
bestmodel = goodmodels[idx]
print("Best Model of My Project: ", bestmodel)

# test set에 대한 측정
cls = bestmodel.classes_
accuracy, precision, recall, f1 = evaluate_model(x_train, x_test, y_train, y_test, bestmodel.predict(x_test), bestmodel, cls)
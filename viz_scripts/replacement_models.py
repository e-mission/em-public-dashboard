import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def gbdt(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
#     # Scale numeric variables
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data[feature_list])
    data_scaled = data[feature_list].values

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data_scaled[train_indices], data_scaled[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        model = GradientBoostingClassifier(n_estimators=50)
        model.fit(X_train, y_train)

        # Predict for test set
        y_pred = model.predict(X_test)
        
        # Track metrics
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return model, accuracy, f1, confusion

def rf(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
#     # Scale numeric variables
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data[feature_list])
    data_scaled = data[feature_list].values

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data_scaled[train_indices], data_scaled[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)

        # Predict for test set
        y_pred = model.predict(X_test)
        
        # Track metrics
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return model, accuracy, f1, confusion

def svm(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
    # Scale numeric variables
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[feature_list])

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data_scaled[train_indices], data_scaled[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        model = SVC()
        model.fit(X_train, y_train)

        # Predict for test set
        y_pred = model.predict(X_test)
        
        # Track metrics
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return model, accuracy, f1, confusion

def knn(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
    # Scale numeric variables
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[feature_list])

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data_scaled[train_indices], data_scaled[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        model = KNeighborsClassifier(n_neighbors=len(pd.unique(y_train)))
        model.fit(X_train, y_train)

        # Predict for test set
        y_pred = model.predict(X_test)
        
        # Track metrics
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return model, accuracy, f1, confusion
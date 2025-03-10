import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from classifiers import *

@st.cache_data  # Changed from @st.cache
def get_dataset(dataset_name):
    '''
    Returns data and info
    '''
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    Y = data.target
    classes = data.target_names
    descr = data.DESCR
    return X, Y, classes, descr

@st.cache_data  # Changed from @st.cache
def get_data_info(dataset_name):
    '''
    Returns dataset information
    '''
    if dataset_name == "Iris":
        data = pd.DataFrame(datasets.load_iris().data, columns=datasets.load_iris().feature_names)
    elif dataset_name == "Breast Cancer":
        data = pd.DataFrame(datasets.load_breast_cancer().data, columns=datasets.load_breast_cancer().feature_names)
    else:
        data = pd.DataFrame(datasets.load_wine().data, columns=datasets.load_wine().feature_names)

    shp = data.shape
    cols = data.columns
    null_vals = data.isnull().sum().sum()
    head = data.head()
    return shp, cols, null_vals, head


def create_plot(dataset_name):
    '''
    Creates scatter plot
    '''
    if dataset_name == "Iris":
        data = pd.DataFrame(datasets.load_iris().data, columns=datasets.load_iris().feature_names)
        target = datasets.load_iris().target
    elif dataset_name == "Breast Cancer":
        data = pd.DataFrame(datasets.load_breast_cancer().data, columns=datasets.load_breast_cancer().feature_names)
        target = datasets.load_breast_cancer().target
    else:
        data = pd.DataFrame(datasets.load_wine().data, columns=datasets.load_wine().feature_names)
        target = datasets.load_wine().target

    cols = data.columns
    if len(cols) < 5:
        fig = plt.figure(figsize=(15, 7))
    elif 5 < len(cols) <= 15:
        fig = plt.figure(figsize=(15, 11))
    else:
        fig = plt.figure(figsize=(15, 20))
    for sp in range(0,len(cols)):
        if len(cols) < 5:
            ax = fig.add_subplot(2,3,sp+1)
        elif 5 < len(cols) <= 15:
            ax = fig.add_subplot(5,3,sp+1)
        else:
            ax = fig.add_subplot(10,3,sp+1)
        sns.scatterplot(data=data,
                       x = np.arange(0, data.shape[0]),
                       y = data.loc[:,cols[sp]],
                       hue = target,
                       ax=ax)
        ax.set_xlabel('No. of Records')
        ax.set_ylabel(cols[sp])
    return fig, data


def corr_plot(X):
    '''
    Creates correlation plot
    '''
    fig,ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(X.corr(), vmin=-1.0, vmax=1.0, annot=True,
                linewidths=.5, fmt= '.1f',ax=ax)
    return fig


def add_clf_data(dataset_name):
    '''
    adds dataset and classifier UI
    '''
    if dataset_name in ["Wine","Iris"]:
        clf_name = st.sidebar.selectbox('Select Classifier',
                                        ('KNN', 'Naive Bayes',
                                        'Decision Tree', 'Random Forest'))
    else:
        clf_name = st.sidebar.selectbox('Select Classifier',
                                        ('KNN', 'Naive Bayes', 'SVM',
                                        'Decision Tree', 'Random Forest'))
    # Create parameters UI
    if clf_name != "Naive Bayes":
        st.sidebar.subheader("Set model parameters")
    return clf_name


def add_parameters(clf_name):
    '''
    Adds parameters, UI and returns parameters
    '''
    params = {}
    if clf_name == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        params["k"] = k
    elif clf_name == "SVM":
        lambd = st.sidebar.slider("Lambda (for model from scratch)", 0.1, 1.0, step=0.05)
        params["lambd"] = lambd
        C = st.sidebar.slider("C (for scikit-learn model)", 1, 10, step=1)
        params["C"] = C
    elif clf_name == "Decision Tree":
        min_split_samples = st.sidebar.slider("Min. split samples", 2, 10, step=1)
        params["min_split_samples"] = min_split_samples
        max_depth = st.sidebar.slider("Max. depth", 10, 100, step=5)
        params["max_depth"] = max_depth
    elif clf_name == "Random Forest":
        n_trees = st.sidebar.slider("No. of trees", 50, 500, step=50)
        params["n_trees"] = n_trees
        min_split_samples = st.sidebar.slider("Min. split samples", 2, 10, step=1)
        params["min_split_samples"] = min_split_samples
        max_depth = st.sidebar.slider("Max. depth", 10, 100, step=5)
        params["max_depth"] = max_depth
    return params


def standardize_data(clf_name, x_train, x_test):
    '''
    Standardizes train and test data
    '''
    if clf_name in ['KNN', 'SVC', 'Naive Bayes']:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
    return x_train, x_test


def get_classifiers(clf_name, params):
    '''
    Builds classifier objects and returns classifiers along with parameters
    '''
    if clf_name == "KNN":
        my_clf = My_KNN()
        param = params["k"]
        skt_clf = KNeighborsClassifier(n_neighbors=param)
    elif clf_name == "Naive Bayes":
        my_clf = NaiveBayes()
        skt_clf = GaussianNB()
        param = None
    elif clf_name == "SVM":
        my_clf = My_SVC()
        param = params["lambd"]
        skt_clf = SVC(C=params["C"], kernel='linear')
    elif clf_name == "Decision Tree":
        my_clf = My_Tree()
        param = []
        param.append(params["min_split_samples"])
        param.append(params["max_depth"])
        skt_clf = DecisionTreeClassifier(max_depth=params["max_depth"],
                                        min_samples_split=params["min_split_samples"])
    elif clf_name == "Random Forest":
        my_clf = My_RandomForest()
        param = []
        param.append(params["n_trees"])
        param.append(params["min_split_samples"])
        param.append(params["max_depth"])
        skt_clf = RandomForestClassifier(n_estimators=params["n_trees"],
                                        max_depth=params["max_depth"],
                                        min_samples_split=params["min_split_samples"],
                                        bootstrap=True)
    return my_clf, skt_clf, param


def skt_clf_fit(skt_clf,x_train,y_train,x_test,y_test):
    '''
    Fits scikit-learn models
    '''
    skt_clf.fit(x_train, y_train)
    y_pred_skt = skt_clf.predict(x_test)
    # Model results
    acc_skt = np.round(accuracy_score(y_test, y_pred_skt), 4)
    test_report_skt = classification_report(y_test, y_pred_skt)
    wt_avg_str_skt = test_report_skt.split('weighted avg')[1].strip().split('      ')
    precision_skt = float(wt_avg_str_skt[1])
    recall_skt = float(wt_avg_str_skt[2])
    cm_skt = confusion_matrix(y_test, y_pred_skt)
    cls_skt = set(y_test)
    cm_skt_fig = plot_cm(cm_skt, cls_skt)
    return acc_skt,precision_skt,recall_skt,cm_skt_fig


def scratch_clf_fit(clf_name,my_clf,x_train,y_train,x_test,y_test,param):
    '''
    Fits models created from scratch
    '''
    if clf_name == "KNN":
        my_clf.fit(x_train, y_train, k=param)
    elif clf_name == "Naive Bayes":
        my_clf.fit(x_train, y_train)
    elif clf_name == "SVM":
        y_train = np.where(y_train <= 0, -1, 1)
        my_clf.fit(x_train, y_train, learning_rate=0.001, epochs=5, lambd=param)
        y_test = np.where(y_test <= 0, -1, 1)
    elif clf_name == "Decision Tree":
        my_clf.fit(x_train, y_train, max_depth=param[0], min_split_samples=param[1])
    elif clf_name == "Random Forest":
        my_clf.fit(x_train, y_train, n_trees=param[0],
                    max_depth=param[1], min_split_samples=param[2],
                    n_feats=3, replace=True)
    # Model results
    y_pred = my_clf.predict(x_test)
    acc = np.round(accuracy_score(y_test, y_pred), 4)
    test_report = classification_report(y_test, y_pred)
    wt_avg_str = test_report.split('weighted avg')[1].strip().split('      ')
    precision = float(wt_avg_str[1])
    recall = float(wt_avg_str[2])
    cm = confusion_matrix(y_test, y_pred)
    cls = set(y_test)
    cm_fig = plot_cm(cm, cls)
    return acc,precision,recall,cm_fig


def plot_cm(cm, cls):
    '''
    Plots confusion matrix
    '''
    fig = plt.figure(figsize=(2,2))
    ax = plt.axes()
    sns.heatmap(cm, square=True, annot=True, annot_kws={'fontsize':8},
                fmt='d', cbar=False, xticklabels=cls, yticklabels=cls, ax=ax)
    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    return fig


def find_best_k(x_train, y_train, x_test, y_test, highest_k=15):
    '''
    Iterates over different values of k, computes error and plot error for each k
    '''
    errors = []

    for i in range(1,highest_k):
        model = My_KNN()
        model.fit(x_train, y_train, k=i)
        pred = model.predict(x_test)
        loss = np.mean(pred != y_test)
        errors.append(loss)

    # Plot errors
    fig = plt.figure(figsize=(10,6))
    plt.plot(range(1,highest_k), errors, color='blue',
             linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K-Value')
    plt.xlabel('K-Value')
    plt.ylabel('Error Rate')
    return fig

def visualize_tree(clf_name, skt_clf, columns, class_names):
    '''
    Plots decision tree
    '''
    if clf_name == "Random Forest":
        clf = skt_clf.estimators_[0]
    else:
        clf = skt_clf
    fig = plt.figure(figsize=(15,10))
    u = tree.plot_tree(clf,
                       feature_names=columns,
                       class_names=class_names,
                       label="root",
                       filled=True)
    return fig

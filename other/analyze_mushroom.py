import os
import pandas as pd
import numpy as np
import time
# import MidpointNormalize

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import *

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def preprocessing_data(filename):
    data = pd.read_csv(filename)
    X = data.drop('category', axis=1)
    X = X.drop('V11', axis=1)
    y = data.category
    return X, y

def tune_kNN(X, y):
    print '-------------------------hyperparameter tuning for k-nearest neighbors-------------------------'

    # folds = 5
    folds = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # using validation_curve to find the optimal k value
    k_range = range(1, 11)
    train_scores, cv_scores = validation_curve(KNeighborsClassifier(), X, y, param_name='n_neighbors',
                                               param_range=k_range, cv=folds, scoring='accuracy', n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    optimal_k = 0
    highest_score = 0
    for i in range(len(k_range)):
        print 'k=%d, training_score: %f, cv_score: %f' % (k_range[i], train_scores_mean[i], cv_scores_mean[i])
        if cv_scores_mean[i] > highest_score:
            highest_score = cv_scores_mean[i]
            optimal_k = k_range[i]
    print 'model selected: %d-nearest neighbors' % (optimal_k)
    # print train_scores_mean, cv_scores_mean

    plot_curve(k_range, train_scores_mean, train_scores_std, cv_scores_mean, cv_scores_std,
               title='Validation Curve with k-nearest neighbors', x_label='k',
               x_lim=(0, 11), y_lim=(0.95, 1.05))

    # generates learning curve

    train_sizes = np.linspace(.1, 1.0, 5)
    # print train_sizes.shape

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(
        KNeighborsClassifier(optimal_k), X, y, cv=folds, n_jobs=4, scoring='accuracy', train_sizes=train_sizes)

    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])
    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with k-nearest neighbors (k=%d)' % (optimal_k), x_label='number of samples',
               x_lim=(0, 6000), y_lim=(0.95, 1.05))

    return {'n_neighbors': optimal_k}

def tune_SVM_1(X, y):
    print '-------------------------hyperparameter tuning for linear SVM-------------------------'

    # folds = 5
    folds = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # c_range = [0.001, 0.01, 0.1]

    train_scores, cv_scores = validation_curve(SVC(kernel='linear'), X, y, param_name='C',
                                               param_range=c_range, cv=folds, scoring='accuracy', n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    optimal_c = 0.0
    highest_score = 0
    for i in range(len(c_range)):
        print 'c=%f, training_score: %f, cv_score: %f' % (c_range[i], train_scores_mean[i], cv_scores_mean[i])
        if cv_scores_mean[i] > highest_score:
            highest_score = cv_scores_mean[i]
            optimal_c = c_range[i]
    print 'optimized value of C: %f' % (optimal_c)
    # print train_scores_mean, cv_scores_mean

    plot_curve(np.log10(c_range), train_scores_mean, train_scores_std, cv_scores_mean, cv_scores_std,
               title='Validation Curve with SVM (linear)', x_label='value of log10(C)',
               x_lim=(-4, 4), y_lim=(0.95, 1.05))

    # optimal_c = 100.0

    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(SVC(kernel='linear'), X, y, cv=folds,
                                                                n_jobs=4, scoring='accuracy', train_sizes=train_sizes)

    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])

    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with SVM (linear, c=%5.2f)' % (optimal_c), x_label='number of samples',
               x_lim=(0, 6000), y_lim=(0.95, 1.05))

    return {'C': optimal_c}

def tune_SVM_2(X, y):
    print '-------------------------hyperparameter tuning for RBF SVM-------------------------'

    folds = 5
    cv_s = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # c_range = [0.001,0.01, 0.1, 1, 10, 100, 1000]
    c_range = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    gamma_range = np.logspace(-5, 1, 7)
    #print gamma_range

    param_grid = dict(gamma=gamma_range, C=c_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv_s, scoring='accuracy', n_jobs=4)
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(c_range), len(gamma_range))
    print scores
    print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

    # Draw heatmap of the cross validation accuracy as a function of gamma and C
    plot_heatmap(scores, gamma_range, c_range, vmin=0.2, mid=0.8, xlabel='gamma', ylabel='C',
                 title='Cross Validation accuracy score with RBF SVM')

    # plot learning curve
    optimal_gamma = grid.best_params_['gamma']
    optimal_c = grid.best_params_['C']
    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(SVC(gamma=optimal_gamma, C=optimal_c), X, y, cv=cv_s,
                                                                n_jobs=4, scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])

    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with SVM (RBF, gamma= %.2f, c=%.2f)' % (optimal_gamma, optimal_c),
               x_label='number of samples', x_lim=(0, 6000), y_lim=(0.95, 1.05))

    return {'gamma': optimal_gamma, 'C': optimal_c}

    """
    train_scores, cv_scores = validation_curve(SVC(gamma = 1), X, y, param_name='C',
                                                 param_range=c_range, cv=folds, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    optimal_c = 0.0
    highest_score = 0
    for i in range(len(c_range)):
        print 'c=%6.2f, training_score: %f, cv_score: %f' % (c_range[i], train_scores_mean[i], cv_scores_mean[i])
        if cv_scores_mean[i] > highest_score:
            highest_score = cv_scores_mean[i]
            optimal_c = c_range[i]
    print 'optimized value of C: %f' % (optimal_c)
    print train_scores_mean, cv_scores_mean

    plot_curve(np.log10(c_range), train_scores_mean, train_scores_std, cv_scores_mean, cv_scores_std,
               title='Validation Curve with SVM (RBF)', x_label='value of log10(C)',
               x_lim=(-4, 4), y_lim=(0.75, 1.0))

    """

def tune_decisionTree(X, y):
    print '-------------------------hyperparameter tuning for decision tree classification-------------------------'

    # folds = 5
    folds = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # using validation_curve to find the optimal k value
    depth_range = range(2, 30)
    train_scores, cv_scores = validation_curve(DecisionTreeClassifier(max_features='auto'), X, y,
                                               param_name='max_depth',
                                               param_range=depth_range, cv=folds, scoring='accuracy', n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    optimal_max_depth = 0
    highest_score = 0
    for i in range(len(depth_range)):
        print 'k=%d, training_score: %f, cv_score: %f' % (depth_range[i], train_scores_mean[i], cv_scores_mean[i])
        if cv_scores_mean[i] > highest_score:
            highest_score = cv_scores_mean[i]
            optimal_max_depth = depth_range[i]
    print 'model selected: %d max depth' % (optimal_max_depth)
    # print train_scores_mean, cv_scores_mean

    plot_curve(depth_range, train_scores_mean, train_scores_std, cv_scores_mean, cv_scores_std,
               title='Validation Curve with decision tree', x_label='max depth',
               x_lim=(0, 35), y_lim=(0.70, 1.05))

    # generates learning curve
    train_sizes = np.linspace(.1, 1.0, 5)  # print train_sizes.shape

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(
        DecisionTreeClassifier(max_features='auto', max_depth=optimal_max_depth),
        X, y, cv=folds, n_jobs=4, scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])
    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with decision tree (max depth = %d)' % (optimal_max_depth),
               x_label='number of samples', x_lim=(0, 6000), y_lim=(0.95, 1.05))

    return {'max_depth': optimal_max_depth}

def tune_boost(X, y):
    print '-------------------------hyperparameter tuning for Adaboosting-------------------------'

    folds = 5
    cv_s = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    n_estimators_range = range(1, 302, 50)
    rate_range = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    print n_estimators_range
    print rate_range

    param_grid = dict(n_estimators=n_estimators_range, learning_rate=rate_range)
    base = DecisionTreeClassifier(max_depth=1)
    grid = GridSearchCV(AdaBoostClassifier(base_estimator=base), param_grid=param_grid, cv=cv_s, scoring='accuracy', n_jobs=4)
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(rate_range), len(n_estimators_range))
    # print scores
    print("The best parameters are %s with an accuracy of %0.5f" % (grid.best_params_, grid.best_score_))

    # Draw heatmap of the cross validation accuracy as a function of gamma and C
    plot_heatmap(scores, n_estimators_range, rate_range, vmin=0.8, mid=0.95, xlabel='n_estimators',
                 ylabel='learning_rate', title='Cross Validation accuracy score with Adaboosting')

    # plot learning curve
    optimal_n_estimators = grid.best_params_['n_estimators']
    optimal_lr = grid.best_params_['learning_rate']
    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(
        AdaBoostClassifier(base_estimator=base, n_estimators=optimal_n_estimators, learning_rate=optimal_lr),
        X, y, cv=cv_s, n_jobs=4, scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])

    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with Adaboosting(%d of estimators, %.3f learning rate)' % (
                   optimal_n_estimators, optimal_lr), x_label='number of samples', x_lim=(0, 6000), y_lim=(0.75, 1.05))

    return {'n_estimators': optimal_n_estimators, 'learning_rate': optimal_lr}

def tune_NN(X, y):
    print '-------------------------hyperparameter tuning for Neural Networks-------------------------'

    folds = 5
    cv_s = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    hidden_layer_sizes_range = [(1,), (5,),(10,), (20,), (100,)]
    n_hidden_units = [1, 5, 10, 20, 100]
    #activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    print hidden_layer_sizes_range
    print alpha_range

    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range, alpha=alpha_range)
    grid = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=cv_s, scoring='accuracy', n_jobs=4)
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(alpha_range), len(hidden_layer_sizes_range))
    # print scores
    print("The best parameters are %s with an accuracy of %0.5f" % (grid.best_params_, grid.best_score_))

    # Draw heatmap of the cross validation accuracy as a function of gamma and C
    plot_heatmap(scores, n_hidden_units, alpha_range, vmin=0.95, mid=0.975, xlabel='n_hidden_units',
                 ylabel='alpha', title='Cross Validation accuracy score with Neuron Network')

    # plot learning curve
    optimal_hls = grid.best_params_['hidden_layer_sizes']
    optimal_alpha = grid.best_params_['alpha']
    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores_lc, cv_scores_lc = learning_curve(
        MLPClassifier(hidden_layer_sizes=optimal_hls, alpha=optimal_alpha),
        X, y, cv=folds, n_jobs=4, scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
    train_scores_std_lc = np.std(train_scores_lc, axis=1)
    cv_scores_mean_lc = np.mean(cv_scores_lc, axis=1)
    cv_scores_std_lc = np.std(cv_scores_lc, axis=1)

    for i in range(train_sizes.shape[0]):
        print 'sample_size=%d, training_score: %f, cv_score: %f' % (
            train_sizes[i], train_scores_mean_lc[i], cv_scores_mean_lc[i])

    # print train_scores_mean_lc, cv_scores_mean_lc

    plot_curve(train_sizes, train_scores_mean_lc, train_scores_std_lc, cv_scores_mean_lc, cv_scores_std_lc,
               title='Learning Curve with Neural Network(%d hidden units and %.5f alpha)' % (
                   optimal_hls[0], optimal_alpha), x_label='number of samples', x_lim=(0, 6000), y_lim=(0.75, 1.05))

    return {'hidden_layer_sizes': optimal_hls, 'alpha': optimal_alpha}

def plot_heatmap(scores, x_range, y_range, **kwargs):
    # Draw heatmap of the cross validation accuracy as a function of two hyperparameters
    # adapted from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=kwargs['vmin'], midpoint=kwargs['mid']))
    plt.xlabel(kwargs['xlabel'], fontsize=12)
    plt.ylabel(kwargs['ylabel'], fontsize=12)
    plt.colorbar()
    plt.xticks(np.arange(len(x_range)), x_range)
    plt.yticks(np.arange(len(y_range)), y_range)
    plt.title(kwargs['title'])
    figure_name = kwargs['title'] + '.png'
    plt.savefig(os.path.join('exported figures/mushroom', figure_name), dpi=72)
    plt.show()

def plot_curve(x_data, train_score_mean, train_score_std, cv_score_mean, cv_score_std, **kwargs):
    plt.title(kwargs['title'])
    plt.xlabel(kwargs['x_label'], fontsize=12)
    plt.xlim(kwargs['x_lim'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(kwargs['y_lim'])

    plt.errorbar(x_data, train_score_mean, yerr=train_score_std, label='Training Score',
                 color='red', linewidth=2, elinewidth=0.5, capsize=4)

    plt.errorbar(x_data, cv_score_mean, yerr=cv_score_std, label='Cross Validation Score',
                 color='green', linewidth=2, elinewidth=0.5, capsize=4)

    plt.legend(loc='lower right')
    figure_name = kwargs['title'] + '.png'
    plt.savefig(os.path.join('exported figures/mushroom', figure_name), dpi=72)
    plt.show()

def evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test):
    for clf in classifiers:

        start_learn_time = time.clock()
        clf.fit(X_train, y_train)
        end_learn_time = time.clock()
        learn_time = end_learn_time - start_learn_time

        start_pred_time = time.clock()
        y_pred = clf.predict(X_test)
        end_pred_time = time.clock()
        pred_time = end_pred_time - start_pred_time

        score_auc = roc_auc_score(y_test, y_pred)
        score_accuracy = accuracy_score(y_test, y_pred)
        score_f1 = f1_score(y_test, y_pred)
        print "---------------------------------------------------------------------------------------"
        print(clf)
        print 'TEST: roc_auc_score: %.5f, accuracy: %.5f, F-score: %.5f. '%(score_auc, score_accuracy, score_f1)
        print 'LEARN_TIME: %s; PREDICT_TIME: %s' % (learn_time, pred_time)

if __name__ == '__main__':
    dataset2 = "agaricus-lepiota.csv"
    X, y = preprocessing_data(dataset2)
    #print X.shape
    X_onehot = pd.get_dummies(X)
    #print X.shape
    #print y.shape
    #print y.head(10)
    y_binary = LabelBinarizer().fit_transform(y).reshape(len(y))
    #print y.shape

    X_train, X_test, y_train, y_test = train_test_split(X_onehot, y_binary, test_size=0.3, random_state=0)

    #para_kNN = tune_kNN(X_train, y_train)
    #para_tree = tune_decisionTree(X_train, y_train)
    #para_linear_SVM = tune_SVM_1(X_train, y_train)
    #para_RBF_SVM = tune_SVM_2(X_train, y_train)
    #para_boost = tune_boost(X_train ,y_train)
    #para_NN = tune_NN(X_train, y_train)

    classifiers = [
        KNeighborsClassifier(n_neighbors=1),
        DecisionTreeClassifier(max_features='auto', max_depth=16),
        SVC(kernel='linear', C=1),
        SVC(C=10, gamma=0.1),
        MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,)),
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.1, n_estimators=250)
    ]

    evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test)









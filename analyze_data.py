import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def preprocessing_data(filename):
    data = pd.read_csv(filename)
    X = data.drop('category', axis = 1)
    y = data.category
    return X, y

def tune_kNN(X, y):

    print '-------------------------hyperparameter tuning for k-nearest neighbors-------------------------'

    k_range = range(1,11)
    train_scores, cv_scores = validation_curve(KNeighborsClassifier(), X, y, param_name='n_neighbors',
                                               param_range=k_range,cv=10, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    optimal_k = 0
    highest_score = 0

    for i in k_range:
        print 'k=%d, training_score: %f, cv_score: %f'%(i, train_scores_mean[i-1], cv_scores_mean[i-1])
        if cv_scores_mean[i-1] > highest_score:
            highest_score = cv_scores_mean[i-1]
            optimal_k = i

    print 'model selected: %d-nearest neighbors' % (optimal_k)

    #print train_scores_mean, cv_scores_mean

    plot_validation_curve(k_range, train_scores_mean, train_scores_std, cv_scores_mean,cv_scores_std,
                          title='Validation Curve with k-nearest neighbors', x_label='k', x_lim = (0, 11),
                          y_lim=(0.8, 1.1) )


def plot_validation_curve(x_data, train_score_mean, train_score_std, cv_score_mean, cv_score_std, **kwargs):

    plt.title(kwargs['title'])
    plt.xlabel(kwargs['x_label'])
    plt.xlim(kwargs['x_lim'])
    plt.ylabel('Accuracy')
    plt.ylim(kwargs['y_lim'])

    plt.errorbar(x_data, train_score_mean, yerr=train_score_std, label='Training Score',
             color='red', linewidth=2, elinewidth=0.5, capsize=4)

    plt.errorbar(x_data, cv_score_mean, yerr=cv_score_std, label='Cross Validation Score',
             color='green', linewidth=2, elinewidth=0.5, capsize=4)

    plt.legend(loc='lower left')
    figure_name = kwargs['title'] + '.png'
    plt.savefig(os.path.join('exported figures', figure_name), dpi=72)
    plt.show()



if __name__ == '__main__':
    dataset1 = "HTRU_2.csv"
    X, y = preprocessing_data(dataset1)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    #print (scaler.mean_)
    #print X.values[0:1, :]
    #print X_scaled[0:1, :]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state = 0)

    """
    print X_train.shape, X_test.shape
    pos_count = 0
    for i in y_train:
        if i == 1:
            pos_count += 1
    print float(pos_count)/y_train.shape[0]
    """

    tune_kNN(X_train, y_train)


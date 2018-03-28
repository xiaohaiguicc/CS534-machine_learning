import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

def SVM(param_grid,cv,kernel):
    '''
    :param param_grid: parameter in SVM grid
    :param cv : cv-fold
    :param kenel: string, thr name of kernel
    :return: svm grid 
    '''
    grid = GridSearchCV(SVC(kernel=kernel),param_grid,cv = cv)
    return grid

def Best_estimator(grid,X,Y):
    '''
    :param grid: SVM grid
    :param X : Training data X
    :param Y : Training data Y
    :return: best estimator after fiting training data. It has smallest error and best parameter(gamma or degree)
    '''
    grid.fit(X, Y)
    best_estimator = grid.best_estimator_
    print("The best classifier is: ", best_estimator)
    return best_estimator

def plot(grid,C_range,para_range,s):
    '''
    :param grid: SVM grid
    :param C_range : range of C parameter
    :param para_range : range of gamma or degree parameter
    :param s : string, name of parameter('gamma' or 'degree')
    :return: plot of score vs parameter
    '''
    scores = [x[1] for x in grid.grid_scores_]

    #plot
    plt.plot(para_range, scores)
    
    plt.legend()
    plt.xlabel(s)
    plt.ylabel('Mean score')
    plt.show()

def Error_rate(X,Y,X2,Y2,best_estimator):
    '''
    :param best_estimator: best_estimator with best parameter
    :param X: list (n*20), training data set in all axis (X[xi,x2])
    :param Y: list (n*1), training data, all data labels
    :param X2: list (n*20), testing data set in all axis (X[xi,x2])
    :param Y2: list (n*1), testing data, all data labels
    :return: error_rate
    '''
    s = X2.shape[0]
    n = 0
    best_estimator.fit(X,Y)
    Y_p = best_estimator.predict(X2)
    for i, y in enumerate(Y_p):
        if (y != Y2[i]):#error_rate = #points whose predict class is different form original one/#total points
            n += 1
    return n/float(s)   


#read files
f = pd.read_csv('/Users/ccai28/Desktop/hw2_data_2.txt',sep='\t', header = None, skiprows = 1)

#read training data
X = f.loc[:699,:19].values
Y = f.loc[:699,20].values

#read testing data
X_test = f.loc[700:,:19].values
Y_test = f.loc[700:,20].values


#set parameter for Gridsearch
C_range = [1]
gamma_range = np.arange(0.01, 0.8,0.01)
degree_range = np.arange(0, 10,1)

param_grid_g = dict(gamma=gamma_range)
param_grid_d = dict(degree=degree_range)
cv = 10

#radical kernel
grid_r = SVM(param_grid_g,cv,'rbf')
best_estimator_r = Best_estimator(grid_r,X,Y)
plot(grid_r,C_range,gamma_range,'gamma')

error_rate_r = Error_rate(X,Y,X_test,Y_test,best_estimator_r)
print('radical kernel: \nerror rate = {:f} \n'.format(error_rate_r))


#sigmoid kernel
grid_s = SVM(param_grid_g,cv,'sigmoid')
best_estimator_s = Best_estimator(grid_s,X,Y)
plot(grid_s,C_range,gamma_range,'gamma')

error_rate_s = Error_rate(X,Y,X_test,Y_test,best_estimator_s)
print('sigmoid kernel: \nerror rate = {:f} \n'.format(error_rate_s))

# polynomial kernel
grid_p = SVM(param_grid_d,cv,'poly')
best_estimator_p = Best_estimator(grid_p,X,Y)
plot(grid_p,C_range,degree_range,'degree')

error_rate_p = Error_rate(X,Y,X_test,Y_test,best_estimator_p)
print('polynomial kernel: \nerror rate = {:f} \n'.format(error_rate_p))

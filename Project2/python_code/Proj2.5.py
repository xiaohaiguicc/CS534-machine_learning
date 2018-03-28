import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

def Error_rate(X_test,Y_test,grid):
    '''
    :param grid: grid after fit by training data
    :param X_test:  (n*20),testing data set in all axis (X[xi,x2])
    :param Y_test:  (n*1),testing data, all data labels
    :return: error_rate
    '''
    s = X_test.shape[0]
    n = 0
    Y_p = grid.predict(X_test)
    for i in range(s):
        if (Y_p[i] != Y_test[i]):#error_rate = #points whose predict class is different form original one/#total points
            n += 1
    score = grid.score(X_test,Y_test)
    return 1-score 

#read files
f = pd.read_csv('/Users/ccai28/Desktop/hw2_data_2.txt',sep='\t', header = None, skiprows = 1)

#read training data
X_train = f.loc[:699,:19].values
Y_train = f.loc[:699,20].values

#read testing data
X_test = f.loc[700:,:19].values
Y_test = f.loc[700:,20].values

#use gradient boosting with deviance loss 
grid = GradientBoostingClassifier(loss = 'deviance')
grid.fit(X_train,Y_train)

#importance and feature
importances = grid.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
error_rate = Error_rate(X_test,Y_test,grid)
print('gradient boosting: \nerror rate = {:f} \n'.format(error_rate))



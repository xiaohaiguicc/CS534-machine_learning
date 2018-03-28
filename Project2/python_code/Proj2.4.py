import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


def Error_rate(X2,Y2,forest):
    '''
    :param forest: forest with best parameter
    :param X2:  (n*20),testing data set in all axis (X[xi,x2])
    :param Y2:  (n*1),testing data, all data labels
    :return: error_rate
    '''
    s = X2.shape[0]
    n = 0
    Y_p = forest.predict(X2)
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
    
# Range of n_estimators values to explore.
min_estimators = 10
max_estimators = 500
step = 10
number_estimators = (max_estimators - min_estimators)/step #how many trees are tried

print number_estimators

error_rate = []
n_estimators = []

#use Random Forest 
#oob_score : bool (default=False) :Whether to use out-of-bag samples to estimate the generalization accuracy.
for i in range(min_estimators, max_estimators + step,step):
    clf = RandomForestClassifier(oob_score=True)#warm_start=True
    clf.set_params(n_estimators=i)
    clf.fit(X, Y)

    # Record the OOB error for each n_estimators=i setting.
    oob_error = 1 - clf.oob_score_
    n_estimators.append(i)
    error_rate.append(oob_error)
    print i,oob_error

last_oob_error = error_rate[number_estimators] #oob error of n = 500

stabilize_estimator = 0
#number of trees based on when the OOB error rate first stabilizes 
for i in range(number_estimators):
    diff_rate = abs((error_rate[i] - last_oob_error)/float(last_oob_error))
    print i, diff_rate
    if (diff_rate <=0.002):
        stabilize_estimator = n_estimators[i] 
        print stabilize_estimator
        break
    
print stabilize_estimator #the stablize tree number

# Generate the "OOB error rate" vs. "n_estimators" plot.
plt.plot(n_estimators, error_rate)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend()
plt.show()

# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators = stabilize_estimator)
forest.fit(X,Y)

#importance and feature
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
error_rate = Error_rate(X_test,Y_test,forest)
print('Random Forest: \nerror rate = {:f} \n'.format(error_rate))

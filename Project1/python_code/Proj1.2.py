from sklearn.neighbors.kde import KernelDensity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(X,x,bandwidth):
    kde_1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)  
    d = 10**(kde_1.score_samples(X)) #using scleaner to calculate density p(x)
    return d

def predict(d1,d2,frac):
    sum = d1*frac+d2*frac
    k = []
    k.append(d1*frac/float(sum))   #calculate posterior according to (density * prior)/evidence
    k.append(d2*frac/float(sum))
    #classify
    new_y = np.argmax(k)           #find the index with 1 -> predict value
    return new_y

def Error_rate(Y,new_Y):
    s = Y.shape[0]
    n = 0
    for i in range(s):
        if (Y[i] != new_Y[i]):         #error_rate = #points of testing predict to another class/#totla points
            n += 1
    return n/float(s)

def misclassified_point(X,Y,new_Y):
    mis_0 = []
    mis_1 = []
    right_0 = []
    right_1 = []
    s = Y.shape[0]
    for i in range(s):
        if (Y[i] != new_Y[i] and new_Y[i] == 0):        #find original class 1 training point -> predict class 0
            mis_0.append(X[i,:])
        elif (Y[i] != new_Y[i] and new_Y[i] == 1):      #find original class 0 training point -> predict class 1
            mis_1.append(X[i,:])
        elif (Y[i] == new_Y[i] and new_Y[i] == 0):      #find original class 0 training point -> predict class 0
            right_0.append(X[i,:])                      #find original class 1 training point -> predict class 1
        else:
            right_1.append(X[i,:])
    mis_0 = np.matrix(mis_0)
    right_0 = np.matrix(right_0)
    mis_1 = np.matrix(mis_1)
    right_1 = np.matrix(right_1)
    return mis_0,mis_1,right_0,right_1

def plot(X,x1,x2,bandwidth,Y,frac):
    
    d1 = kernel(X,x1,bandwidth)
    d2 = kernel(X,x2,bandwidth)
    
    #predict
    s = X.shape[0]
    new_Y = np.zeros((s,1))
    for i in range(s):
        new_Y[i] = predict(d1[i],d2[i],frac)      #calculate predict value for every point in testing file
    
    #error_rate
    print('When bandwidth={:f}, \nerror rate = {:f} \n'.format(bandwidth, Error_rate(Y,new_Y)))
    
    #mis points
    mis_0,mis_1,right_0,right_1 = misclassified_point(X,Y,new_Y)
    
    plt.scatter(right_0[:,0],right_0[:,1],30,
            color='black', marker='o', label='y = 0')
    plt.scatter(right_1[:,0],right_1[:,1],30,
            color='black', marker='x', label='y = 1')
    plt.scatter(mis_0[:,0],mis_0[:,1],30,
            color='red', marker='o', label='y = 0,mis')
    plt.scatter(mis_1[:,0],mis_1[:,1],30,
            color='red', marker='x', label='y = 1,mis')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='lower right',
	       fontsize = 7)
    plt.savefig('bandwidth={:f}.eps'.format(bandwidth),format='eps')
    plt.show()
    return 0

#read files        
f1 = pd.read_csv('/Users/ccai28/Desktop/HW_1_training.txt',sep='\t', header = None, skiprows = 1)
f2 = pd.read_csv('/Users/ccai28/Desktop/HW_1_testing.txt',sep='\t', header = None, skiprows = 1)

#read testing data
X = f2.loc[:,[0,1]].values
Y = f2.loc[:,[2]].values

#read traing data and kernel density estimator
# y = 0
x1 = f1.loc[lambda f : f[2] == 0,[0,1]].values
#y = 1
x2 = f1.loc[lambda f : f[2] == 1,[0,1]].values

plot(X,x1,x2,10,Y,0.5)
plot(X,x1,x2,1,Y,0.5)
plot(X,x1,x2,0.1,Y,0.5)

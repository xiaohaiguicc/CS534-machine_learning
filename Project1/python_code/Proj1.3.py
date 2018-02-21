import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def distance(x,y):
    d = math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) #distance between testing point and train point
    return d

def KNN(X,x,y,K):
    s = x.shape[0]
    S = X.shape[0]
    new_Y = np.zeros((S,1))
    for i in range(S):
        d = np.zeros((s,1))
        for j in range(s):
            d[j] = distance(X[i,:],x[j,:]) #distance for all the points respect with one point i in testing file
        index = np.argsort(d,axis = 0)     #sort according to the distance and index include index of distance 
        count = 0                          #from smallest to longest
        for k in range(K):    #find k nearest points in training respect to testing i
            y_1 = y[index[k]] #find the class of testing point i with its index
            if y_1 == 0:      #if the nearest test points are in class 0, and more than half of these points are in 
                count += 1    # class 0, the i training point is ->predict into class 0
        if count > K*0.5:     # else ->predict into class 1
            new_y = 0
        else:
            new_y = 1
        new_Y[i] = new_y      #build the new class of traiing points after classfied
        
    return new_Y

def misclassified_point(X,Y,new_Y):
    mis_0 = []
    mis_1 = []
    right_0 = []
    right_1 = []
    s = Y.shape[0]
    for i in range(s):
        if (Y[i] != new_Y[i] and new_Y[i] == 0):   #find original class 1 training point -> predict class 0
            mis_0.append(X[i,:])
        elif (Y[i] != new_Y[i] and new_Y[i] == 1): #find original class 0 training point -> predict class 1
            mis_1.append(X[i,:])
        elif (Y[i] == new_Y[i] and new_Y[i] == 0): #find original class 0 training point -> predict class 0
            right_0.append(X[i,:])                 #find original class 1 training point -> predict class 1
        else:
            right_1.append(X[i,:])
    mis_0 = np.matrix(mis_0)
    right_0 = np.matrix(right_0)
    mis_1 = np.matrix(mis_1)
    right_1 = np.matrix(right_1)
    return mis_0,mis_1,right_0,right_1

def plot(X,Y,x,y,K):
    new_Y = KNN(X,x,y,K)
    
    #mis points
    mis_0,mis_1,right_0,right_1 = misclassified_point(X,Y,new_Y)
    
    
    plt.scatter(right_0[:,0],right_0[:,1],30,
            color='black', marker='o', label='healthy controls')
    plt.scatter(right_1[:,0],right_1[:,1],30,
            color='black', marker='x', label='disease cases')
    plt.scatter(mis_0[:,0],mis_0[:,1],30,
            color='red', marker='o', label='healthy controls,mis')
    plt.scatter(mis_1[:,0],mis_1[:,1],30,
            color='red', marker='x', label='disease cases,mis')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(scatterpoints=1,
           loc='lower right',
           ncol=1,
           fontsize=7)
    plt.savefig('k={:d}.eps'.format(K),format='eps')
    plt.show()
   
    return 0

def parameters(X,Y,x,y,K):
    new_Y = KNN(X,x,y,K)
    right0 = [] #True Negtive, TN
    right1 = [] #True Positive, TP
    mis0 = [] #class 0 -> predict class 1 #False Negtive, FN
    mis1 = [] #class 1 -> predict class 0 #False Positive, FP

    for i in range(Y.shape[0]):
        if np.allclose(new_Y[i],0):
            # Class 0
            if np.allclose(new_Y[i],Y[i]): #Correct class0
                right0.append(new_Y[i])
            else:
                mis0.append(new_Y[i])
        else:
            if np.allclose(new_Y[i],Y[i]):
                right1.append(new_Y[i])
            else:
                mis1.append(new_Y[i])
    TN = len(right0)
    TP = len(right1)
    FN = len(mis0)
    FP = len(mis1)
    sensitivity = TP / float((TP + FN)) 
    specificity = TN / float((TN + FP))
    false_discovery_rate = FP / float((FP + TP))
    
    print('When k={:d}, \nsensitivity = {:f},\nspecificity={:f}, \nfalse discovery rate={:f} \n'.format(K,sensitivity,specificity, false_discovery_rate))




#read data
f1 = pd.read_csv('/Users/ccai28/Desktop/HW_1_training.txt',sep='\t', header = None, skiprows = 1)
f2 = pd.read_csv('/Users/ccai28/Desktop/HW_1_testing.txt',sep='\t', header = None, skiprows = 1)

#read testing data
X = f2.loc[:,[0,1]].values
Y = f2.loc[:,[2]].values

#read traing data
x = f1.loc[:,[0,1]].values
y = f1.loc[:,[2]].values

#parameters,k=1
parameters(X,Y,x,y,1)
#plot,k=1
plot(X,Y,x,y,1)

#k=5
parameters(X,Y,x,y,5)
plot(X,Y,x,y,5)

#k=10
parameters(X,Y,x,y,10)
plot(X,Y,x,y,10)

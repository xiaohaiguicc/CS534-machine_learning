import numpy as np
import pandas as pd

#define functions of prediction
def predict(x,w,y):
    '''
    :param x: one row of data samples
    :param w: data weight
    :param y: one row of data labels
    :return: predict label
    '''
    y_p = np.dot(x,w)*y
    return y_p
#define function of perceptron
def perceptron(X,w,Y,lrate):
    '''
    :param X: data samples
    :param w: data weight
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    flag = True
    while(flag):
        total_error = 0       
        for i, x in enumerate(X):
            x = np.insert(x,2,1) # Add bias = 1 in x
            y_p = predict(x,w,Y[i])
            if (y_p <= 0):
                total_error += 1
                w = w + lrate*x*Y[i] #new weight,lrate = learning rate
        if (total_error == 0):
            flag = False #loop until all training data are predict correct
    return w

#prediction of testing data and error rate
def Error_rate(X2,w,Y2):
    '''
    :param X2: Testing data samples
    :param w: data weight
    :param Y: Testing data labels
    :return: error_rate
    '''
    s = Y2.shape[0]
    n = 0
    for i, x in enumerate(X2):
        x2 = np.insert(x,2,1)
        y2_p = predict(x2,w,Y2[i])
        if (y2_p <= 0):#error_rate = #points whose predict class is different form original one/#total points
            n += 1
    return n/float(s)   
    
#read files
f = pd.read_csv('/Users/ccai28/Desktop/hw2_data_1.txt',sep='\t', header = None, skiprows = 1)

#read training data
X1 = f.loc[:69,[0,1]].values
Y1 = f.loc[:69,[2]].values

#read testing data
X2 = f.loc[70:,[0,1]].values
Y2 = f.loc[70:,[2]].values

#set learning rate and initial weight
lrate = 1
w = np.ones(3)

#new weight after training
w = perceptron(X,w,Y,lrate)

#Error rate 
error_rate = Error_rate(X2,w,Y2)

print('perceptron: \nerror rate = {:f} \n'.format(error_rate))

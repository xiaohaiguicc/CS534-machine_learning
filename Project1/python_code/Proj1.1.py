import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#define functions of parameters and decision
def discrtminant_func(f,y):
    mean_vectors = np.zeros((1,2))     #mean vectro and covariance matrix
    covariance = np.zeros((2,2))
    mean_vectors = np.mean(f.loc[lambda f : f[2] == y,[0,1]].values, axis=0)
    covariance = np.cov(f.loc[lambda f : f[2] == y,[0,1]].values.T)
    return mean_vectors, covariance

def decision_function(x,x_m,x_c,prior_probability):
    sig_inv = np.linalg.inv(x_c)         #decision function
    return -.5*np.log( np.linalg.det(x_c))-.5*np.dot(np.dot((x-x_m),sig_inv),(x-x_m).T) + np.log(prior_probability)

def predict(X,m1,c1,m2,c2,frac1,frac2):
    s = X.shape[0]
    new_Y = np.zeros((s,1))
    X_new = np.zeros((s,2))
    j = 0
    l = s-1
    for i in range(s):
        k = []
        k.append(decision_function(X[i,:],m1,c1,frac1))    #result of decision function of class 1
        k.append(decision_function(X[i,:],m2,c2,frac2))    #result of decision function of class 1
        #classify
        new_Y[i] = np.argmax(k)      #find the index that has bigger results(results in which class is the biggest)
        if np.argmax(k) == 0:
            X_new[j,:] = X[i,:]      #rearrange testing points accroding to predict class
            j += 1
        elif new_Y[i] == 1:         #the predict class of points in testing file
            X_new[l,:] = X[i,:]
            l -= 1 
    
    return X_new,new_Y,j,l

def Error_rate(Y,new_Y):
    s = Y.shape[0]
    n = 0
    for i in range(s):
        if (Y[i] != new_Y[i]):      #error_rate = #points whose predict class is different form original one/#total points
            n += 1
    return n/float(s)

def plot(x,m1,c1,m2,c2,frac1,frac2):
    #build grid of xx1,xx2
    xx1,xx2 = np.meshgrid(np.arange(-5, 7,0.02 ),
                         np.arange(-4, 4,0.02))
    #z is the new_Y of xx1 and xx2 field
    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T,m1,c1,m2,c2,frac1,frac2)[1]
    
    #change Z to a matrix with the same shape of xx1
    Z = Z.reshape(xx1.shape)
    #draw the boundary line
    plt.contour(xx1, xx2, Z, alpha=0.3)
    
    #plot points
    plt.scatter(x[:200,0], x[:200,1],30,
            color='red', marker='o', label='y=0')
    plt.scatter(x[200:,0], x[200:,1],30,
    color='blue', marker='x', label='y=1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    return 0


#read data
f1 = pd.read_csv('/Users/ccai28/Desktop/HW_1_training.txt',sep='\t', header = None, skiprows = 1)
f2 = pd.read_csv('/Users/ccai28/Desktop/HW_1_testing.txt',sep='\t', header = None, skiprows = 1)

#mean vector and covariance matrix
x = f1.loc[:,[0,1]].values
m1,c1 = discrtminant_func(f1,0)
m2,c2 = discrtminant_func(f1,1)
print m1,m2,c1,c2
#read testing data
X = f2.loc[:,[0,1]].values
Y = f2.loc[:,[2]].values 

#equal prior
X_new_e,new_Y_e,j_e,l_e = predict(X,m1,c1,m2,c2,0.5,0.5)
error_rate_e = Error_rate(Y,new_Y_e)
print('Equal prior: \nerror rate = {:f} \n'.format(error_rate_e))
plot(x,m1,c1,m2,c2,0.5,0.5)
plt.savefig('Equal prior .eps',format='eps')
plt.show()

#from data
X_new_d,new_Y_d,j_d,l_d = predict(X,m1,c1,m2,c2,200/float(325),125/float(325))
error_rate_d = Error_rate(Y,new_Y_d)
print('Prior calculated from the data: \nerror rate = {:f} \n'.format(error_rate_e))
plot(x,m1,c1,m2,c2,200/float(325),125/float(325))
plt.savefig('Prior calculated from the data.eps',format='eps')
plt.show()

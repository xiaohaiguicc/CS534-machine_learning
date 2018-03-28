import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#define functions of classifier
def lineclassify(value,x,label):
    '''
    :param value: double,the standard value to compare with
    :param x:  training data,double,data axis value
    :label: 0,left; 1,right
    :left : x <= value: predict y is -1; x > value: predict y is 1
    :right: x <= value: predict y is 1; x > value: predict y is -1
    :return: predict y,int(-1 or 1)
    '''
    if (label == 0):
        if (x <= value):
            y_p = -1      
        else:
            y_p = 1
    elif (label == 1):
        if (x <= value):
            y_p = 1      
        else:
            y_p = -1
           
    return y_p


#Classify all training data with one weak classfier (xij)(loop all training data with specific xij)
def Weighterror(value,X,Y,W,label):
    '''
    :param value: double,the standard value to compare with
    :param X:  training data,list (n*1),data set in one axis (eg.X[:,0])
    :param Y:  training data,list (n*1),all data labels
    :param W: list (n*1),weight set of data
    :label: 0,left; 1,right
    :left : x <= value: predict y is -1; x > value: predict y is 1
    :right: x <= value: predict y is 1; x > value: predict y is -1
    :return: double, weighterror = sum(misclassified data's value on one axis * weight)
    '''
    weighterror = 0
    for i, x in enumerate(X):
        y_p = lineclassify(value,x,label)   
        y_dot = y_p*Y[i]
        
        if (y_dot <= 0):      
            weighterror += W[i]
           
    return weighterror

#The sum of weight
def Sumweight(W):
    '''
    :param W: list (n*1),weight set of data (all equal to 1/N),N is number of training data
    :return: double,sum of weight
    '''
    sumweight = 0
    for i, w in enumerate(W):
        sumweight += W[i]
    return sumweight

#new weight after iteration
def changeweight(value,X_line,Y,W,label,a):
    '''
    :param value: double,the best value to compare with
    :param X:  training data,list (n*1),data set in one axis (eg.X[:,0])
    :param Y:  training data,list (n*1),all data labels
    :param W: list (n*1),weight set of data
    :para a: alpha wi = wi*exp(a)
    :label: 0,left; 1,right
    :left : x <= value: predict y is -1; x > value: predict y is 1
    :right: x <= value: predict y is 1; x > value: predict y is -1
    :return: list,new weight
    '''
    for i,x in enumerate(X_line):
        y_p = lineclassify(value,X_line[i],label)
        y_dot = y_p*Y[i]
        
        if (y_dot <= 0): 
        # print W[j],alpha[i]
            W[i] = W[i]*np.exp(a)#calculate new wi
            
        else:
            continue
    return W

#Find the best xij to classify traning data (loop for every possible xij)
def Buildline(X,Y,W):
    '''
    :param X: training data,list (n*2),data set in all axis (X[xi,x2])
    :param Y: training data,list (n*1),all data labels
    :param W: list (n*1),weight set of data
    :return: list,best xij with smallest weighterror and its label in l/r, X1/X2
             x_best = [best value,[l(0)/r(1),X1(0)/X2(1)]
    :label: left:0 ; right:1; X1:0 ; X2:1
    '''
    weight_x1_l = []
    weight_x1_r = []
    weight_x2_l = []
    weight_x2_r = []
    weight_x = []
    x_best = []
    X_list = []
    
    X1 = X[:,0] #all x1
    X2 = X[:,1] #all x2
    
    X_list.append(X1)
    X_list.append(X1)
    X_list.append(X2)
    X_list.append(X2) #X_list = [[X1],[X1],[X2],[X2]]

        
    for i, x1 in enumerate(X1):
        weighterror1_l = Weighterror(X1[i],X1,Y,W,0) #weighterror of X1,left
        weight_x1_l.append(weighterror1_l)
        
        weighterror1_r = Weighterror(X1[i],X1,Y,W,1)
        weight_x1_r.append(weighterror1_r) #weighterror of X1,right
    
    weight_x.append(weight_x1_l) #put (l,x1) in list, index = 0
    weight_x.append(weight_x1_r) #put (r,x1) in list, index = 1
    
    
    for i, x2 in enumerate(X2):
        weighterror2_l = Weighterror(X2[i],X2,Y,W,0) #weighterror of X2,left
        weight_x2_l.append(weighterror2_l)
        weighterror2_r = Weighterror(X2[i],X2,Y,W,1) #weighterror of X2,right
        weight_x2_r.append(weighterror2_r)
    
    
    weight_x.append(weight_x2_l) #put (l,x2) in list, index = 2
    weight_x.append(weight_x2_r) #put (r,x2) in list, index = 3
                                 #weight_x = [[weight_x1_l], [weight_x1_r], [weight_x2_l], [weight_x2_r]]
    
    weight_x_min_list = np.min(weight_x, axis=1) #minimum of every (l/r,x1/x2) list 
                                                #weight_x_min_list = [min(weight_x1_l),min(weight_x1_r),min(weight_x2_l),min(weight_x2_r)]
    
    weight_x_min = min(weight_x_min_list)#minimum in all list, smallest weighterror

    index_list = np.argmin(weight_x_min_list) #index of four lists who has the smallest weighterror
                                             #index_list = the index in weight_x_min_list
                                             #= the index in weight_x = best value with smallest weighterror in which list, (l/r,x1/x2),between 0 and 3   

    label_list = [[0,0],[1,0],[0,1],[1,1]] #build label list to represent all (l/r,x1/x2)
                                           #label_list = [[l,X1],[r,X1],[l,X2],[r,X2]]
    label = label_list[index_list] #find the label(0 or 1) of smallest weighterror
    
    index = weight_x[index_list].index(weight_x_min)#index of best value,0-69

    best_value = X_list[index_list][index]
    
    x_best.append(best_value)
    x_best.append(label) #x_best = [best value,[l(0)/r(1),X1(0)/X2(1)]
                        #best value = x_best[0], l/r = x_best[1][0],X1/X2 = x_best[1][1]
    
    return x_best

#Build Adaboost training
def Adaboost(iteration,X,Y,W,X_test,Y_test):
    '''
    :param iteration: times loop weak lineclassify
    :param X: list (n*2),training data set in all axis (X[xi,x2])
    :param Y: list (n*1),training data, all data labels
    :param X_test: list (n*2),testing data set in all axis (X[xi,x2])
    :param Y_test: list (n*1),testing data, all data labels
    :param W: list (n*1),initial weight set of data (all equal to 1/N),N is number of training data
    :return: list, error rate, contains every iteration error rate
    '''
    error_rate = []
   
    Y_testp = np.zeros(Y_test.shape[0])
    for i in range(iteration):
        n = 0
        x_best = Buildline(X,Y,W)

        value, label_index ,line_index = x_best[0],x_best[1][0],x_best[1][1]  #value, label_index ,line_index = best value, l/r, X1/X2 
        X_line = X[:,line_index] #x1 best: X_line = X[:,0];x2 best: X_line = X[:,1];
        
        weighterror = Weighterror(value,X_line,Y,W,label_index)
        
        sumweight = Sumweight(W)
        error = weighterror/float(sumweight)  #calculate error
        print ("errorm:")
        print error
        
        #print weighterror,sumweight,error
        a = np.log((1-error)/float(error))  #calculate alpha
        #print error,a,value_best[i]
        
        W = changeweight(value,X_line,Y,W,label_index,a) #new weight
        
        s_test = X_test.shape[0] #number of data in X_test
        for j in range(s_test):
            x_test = X_test[:,line_index][j] 
            Y_testp[j] += AdaClassify(a,x_best,x_test)
        
            y_dot = Y_testp[j] *Y_test[j]
            if (y_dot <= 0):      
                n += 1 
        error_rate.append(n/float(s_test)) #error rate of every iteration
        print error_rate[i]
                    
    return error_rate

#Adaboost classify(combination of weak classifier)
def AdaClassify(alpha,value_best,x_test):
    '''
    :param alpha: to build adaboostc classifier: sum(alpha[i]*Gi(x))
    :param value_best: list,= [[best value1,[l(0)/r(1),X1(0)/X2(1)],[[best value2,[l(0)/r(1),X1(0)/X2(1)]]...]
    :param x_test: element ,testing data of one axis
    :return: predict y with alpha
    '''
    value, label_index ,line_index = value_best[0],value_best[1][0],value_best[1][1]  #value, label_index ,line_index = best value, l/r, X1/X2 
    y_testp = lineclassify(value,x_test,label_index)*alpha #adaboost classifier
               
    return y_testp
       
#read files
f = pd.read_csv('/Users/ccai28/Desktop/hw2_data_1.txt',sep='\t', header = None, skiprows = 1)

#read training data
X = f.loc[:69,[0,1]].values
Y = f.loc[:69,[2]].values

#read testing data
X_test = f.loc[70:,[0,1]].values
Y_test = f.loc[70:,[2]].values

#number of training data
s = X.shape[0]

#set initial weight
W = np.full((s,1),1/float(s))


#iteration
iteration = 20

error_rate = Adaboost(iteration,X,Y,W,X_test,Y_test)
print error_rate[iteration-1]

x_plot = list(range(iteration))

plt.plot(x_plot, error_rate)
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.show()

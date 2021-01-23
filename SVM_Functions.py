#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Created by: Dominick Sinopoli

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from mpl_toolkits import mplot3d




# In[2]:

def basic_SVM_test(x,y,trials = 100,gamma = 'scale'):
    kernels = ['rbf','linear','poly','sigmoid']
    scores_basic = np.zeros((trials,len(kernels)))
    for i in range(len(kernels)):
        for j in range(trials):
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = .25)
            SVC_model = SVC(kernel = kernels[i],gamma = gamma)
            SVC_model.fit(x_train,y_train)
            scores_basic[j,i] = SVC_model.score(x_test,y_test)
    for i in range(len(kernels)):
        print('SVM w/ kernel=' + kernels[i] + ' and gamma=' + gamma,np.average(scores_basic[:,i],axis=0)*100)
    return scores_basic

def make_data_set(data_set,x_var,y_var,PF = 67, make_dummy = True):
    x = data_set[x_var]
    y = data_set[y_var]
    if(make_dummy == True):
        y_PF = np.zeros((len(y),1))
        for i in range(len(y)):
            if(y[i] >= PF):
                y_PF[i] = 1
            else:
                y_PF[i]= 0
        y_PF = y_PF.ravel()
    return x,y_PF


# In[ ]:

def c_cross_validation(x,y,c_values,K_folds = 10,kernel = 'linear',gamma = 'scale'):
    K = K_folds
    c = c_values
    N = len(c)
    scores = np.zeros((N,K))
    kf = KFold(n_splits=K_folds)
    for i in range(N):
        SVC_model = SVC(C = c[i],kernel = kernel,gamma = gamma)
        for j, (train, test) in enumerate(kf.split(x)):
            x_train, x_test, y_train, y_test = x.loc[train], x.loc[test], y[train], y[test]
            SVC_model.fit(x_train,y_train)
            scores[i,j] = SVC_model.score(x_test, y_test) 
    scores_avg = scores.mean(axis=1)
    plt.plot(c,scores.mean(axis=1))
    plt.title('Cross-validation Error Curve K=' + str(K_folds))
    plt.xlabel('C Values')
    plt.ylabel('$R^2$')
    plt.show()


# In[ ]:


def SVM_linear_bootstrap(x,y,x_var_num,test_size = .25,samples = 100,c = 1, gamma = 'scale'):
    var = x_var_num
    coefs = np.zeros((1,var))
    intercepts = np.zeros((samples,1))
    for i in range(samples):
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size)
        SVC_model = SVC(kernel = 'linear',gamma = gamma)
        SVC_model.fit(x_train,y_train)
        coef_tmp = SVC_model.coef_
        coefs = np.vstack((coef_tmp,coefs))
        intercepts[i] = SVC_model.intercept_
    coefs_2 = np.delete(coefs,100,axis=0)
    coefs_adv = np.average(coefs_2,axis=0)
    int_adv = np.average(intercepts,axis=0)
    print('Coefficient Adverage:',coefs_adv)
    print('Intercept Adverage:', int_adv)
    return coefs_adv,int_adv


# In[ ]:


def SVM_test(x,y,coefs_adv,int_adv,x_var,N = 100,test_size = .25):
    scores = np.zeros((N,4))
    index=list()
    for z in range(N):
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size)
        x_test = x_test.reset_index(drop = False)
        predicted_score = np.zeros((len(y_test),1))
        for i in range(len(y_test)):
            predicted_score[i] = int_adv 
            for j in range(len(x_var)):
                predicted_score[i] = predicted_score[i] + (coefs_adv[j]*x_test.loc[i,x_var[j]])
            if(predicted_score[i] > 0):
                predicted_score[i] = 1
            else:
                predicted_score[i]= 0
       
        count = 0
        for i in range(len(predicted_score)):
            if(predicted_score[i] != y_test[i]):
                index.append(x_test.loc[i,'index'])
                count = count + 1

        x_test=x_test.drop(['index'],axis=1)        
        scores[z,0] = (1-count/len(y_test))
        SVC_model = SVC(kernel = 'linear',gamma = 'scale')
        SVC_model.fit(x_train,y_train)
        scores[z,1] = SVC_model.score(x_test,y_test)
        SVC_model = SVC(gamma = 'scale')
        SVC_model.fit(x_train,y_train)
        scores[z,2] = SVC_model.score(x_test,y_test)
        linear_reg =  LinearRegression().fit(x_train, y_train)
        scores[z,3] = linear_reg.score(x_test,y_test)
    print('Boosted SVM w/ linear kernel:',np.average(scores[:,0],axis=0)*100)
    print('SVM w/ linear kernel:',np.average(scores[:,1],axis=0)*100)
    print('SVM w/ default kernel (rbf):',np.average(scores[:,2],axis=0)*100)
    print('Linear Regression:',np.average(scores[:,3],axis=0)*100)
    return scores,index


def make_two_trial_graph(scores,labels,trials = 100):
    x = np.linspace(1,trials,trials)
    colors = ['b','g']
    for i in range(len(labels)):
        plt.plot(x,scores[:,i], colors[i],label = labels[i])
    plt.legend(loc='best')
    plt.title(labels[0] +' vs '+ labels[1])
    plt.xlabel('Trial')
    plt.ylabel('Accuracy $R^2$')
    plt.show()

def make_contour_plot_2D(x,y,x_var,intercept,coefficients):
    h = 0.02
    x_min, x_max = x.loc[:,x_var[0]].min() - 1, x.loc[:, x_var[0]].max() + 1
    y_min, y_max = x.loc[:, x_var[1]].min() - 1, x.loc[:, x_var[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    predict_values = np.c_[xx.ravel(),yy.ravel()]
    Z = np.zeros((len(predict_values),1))
    for i in range(len(predict_values)):
        Z[i] = intercept
        for j in range(len(coefficients)):
            Z[i] = Z[i] + (coefficients[j]*predict_values[i,j])
        if(Z[i] > 0):
            Z[i] = 1
        else:
            Z[i]= 0
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(x.loc[:,x_var[0]], x.loc[:,x_var[1]], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel(x_var[0])
    plt.ylabel(x_var[1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xticks(())
    #plt.yticks(())
    plt.title(x_var[1] + ' vs. ' + x_var[0])
    plt.show()


def make_3D(x,y,x_var,int_adv,coefs_adv,h=0.02):
    x_min, x_max = x.loc[:,x_var[0]].min(), x.loc[:, x_var[0]].max()
    y_min, y_max = x.loc[:, x_var[1]].min(), x.loc[:, x_var[1]].max()
    
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    
    x3 = ((-int_adv)-(coefs_adv[0]*xx)-(coefs_adv[1]*yy))/(coefs_adv[2]*4)
    fig = plt.figure(figsize = plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1,projection = '3d')
    #fig = plt.figure()
    #x = plt.axes(projection = '3d')
    ax.scatter3D(x.loc[:,x_var[0]], x.loc[:,x_var[1]],x.loc[:,x_var[2]]*(1/4), c=y, cmap=plt.cm.RdYlBu)
    ax.plot_surface(xx,yy,x3)
    ax.set_xlabel(x_var[0])
    ax.set_ylabel(x_var[1])
    ax.set_zlabel("A&P Score")
    ax.dist = 11
     
    
    ax = fig.add_subplot(1,2,2,projection = '3d')
    ax.scatter3D(x.loc[:,x_var[0]], x.loc[:,x_var[1]],x.loc[:,x_var[2]]*(1/4), c=y, cmap=plt.cm.RdYlBu)
    ax.plot_surface(xx,yy,x3)
    ax.view_init(azim = -180)
    ax.set_xlabel(x_var[0])
    ax.set_ylabel(x_var[1])
    ax.set_zlabel("A&P Score")
    ax.dist =11
    
    plt.suptitle('3D Hyperplane', fontsize=18)
   
    
    plt.show()
    
def make_error_graph(data_set,index,int_adv,coefs_adv,x_var):
    predicted_score = np.zeros((len(index),1))
    w_norm = np.linalg.norm(coefs_adv)
    for i in range(len(index)):
        predicted_score[i] = int_adv 
        for j in range(len(x_var)):
            predicted_score[i] = predicted_score[i] + (coefs_adv[j]*data_set.loc[index[i],x_var[j]])
        predicted_score[i] = predicted_score[i] / w_norm
    counts = np.zeros((1,1))
    counts[0] = -1000
    counts = np.append(counts,np.linspace(-30,30,61))
    counts = np.append(counts,1000)
    cat = np.zeros((62,1))
    z = 0
    for i in range(len(counts)-1):
        count = 0
        for j in range(len(predicted_score)):
                if((predicted_score[j] > counts[i] ) & (predicted_score[j] < counts[i + 1])):
                    count = count + 1             
        cat[z] = count
        z = z + 1
    #plt.plot(np.linspace(-30,30,62),cat/sum(cat))
    #plt.show()
    plt.hist(predicted_score)
    plt.title('Error')
    plt.xlabel('Distance From Hyperplane')
    plt.ylabel('Number of Misclassified')
    plt.show()

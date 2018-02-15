'''
Created on 15-Nov-2017

@author: ASUS
'''
import pandas as pd
import numpy as np
import math as mt
from array import array

def navies(listC1,listC2,testin_matrix_C1_transpose,testin_matrix_C2_transpose,testin_matrix_C1,testin_matrix_C2):
    probC1=listC1.__len__()/(listC1.__len__() + listC2.__len__())
    probC2=listC2.__len__()/(listC1.__len__() + listC2.__len__())
    sigmaC1=np.cov(np.array(listC1),rowvar=False)
    sigmaC2=np.cov(np.array(listC2),rowvar=False)
    sigmaC1_inverse=np.linalg.inv(sigmaC1)
    sigmaC2_inverse=np.linalg.inv(sigmaC2)
    #Calculation of class in which trainig set belong
    mod_C1=np.linalg.det((np.array(sigmaC1)))
    mod_C2=np.linalg.det((np.array(sigmaC2)))
    prob_cart_C1=probC1*(1/(pow(2 * 3.14,1.5)) *(pow(mod_C1,0.5)))*mt.exp(-0.5*np.matmul(np.matmul((testin_matrix_C1_transpose),(sigmaC1_inverse)),(testin_matrix_C1)))
    prob_cart_C2=probC2*(1/(pow(2 * 3.14,1.5)) *(pow(mod_C2,0.5)))*mt.exp(-0.5*np.matmul(np.matmul((testin_matrix_C2_transpose),(sigmaC2_inverse)),(testin_matrix_C2)))
    prob_C1=prob_cart_C1/(prob_cart_C1 + prob_cart_C2 )
    prob_C2=prob_cart_C2/(prob_cart_C1 + prob_cart_C2 )
    print(prob_cart_C1)
    print(prob_cart_C2)
    print(prob_C1)
    print(prob_C2)
    if(prob_C1>prob_C2):
        print("test set belong to C2 class")
    else:
        print("test set belong to C1 class")
if __name__ == '__main__':
    pf=pd.read_csv("Test.csv")
    rows=pf.shape
    listC1=[]
    listC2=[]
    countC1=0
    countC2=0
    for i in range (0,rows[0]):
        if pf.iloc[i,3]=="C1":
            listC1.append([pf.iloc[i,0],pf.iloc[i,1],pf.iloc[i,2]])
            countC1=countC1 + 1
        else:
            listC2.append([pf.iloc[i,0],pf.iloc[i,1],pf.iloc[i,2]])
            countC2=countC2 + 1
    C1_feature_var=[]
    for i in range (0,rows[1]-1):
        C1_feature_var.append(sum(value[i] for value in listC1))
    C2_feature_var=[]
    for i in range (0,rows[1]-1):
        C2_feature_var.append(sum(value[i] for value in listC1))
    C1_feature_var=[x / countC1 for x in C1_feature_var]
    C2_feature_var=[x / countC2 for x in C2_feature_var]
    num_array = list()
    print("enter training set data")
    num = input("Enter how many elements you want:")
    print('Enter numbers in array:')
    for i in range(int(num)):
        n = input("num")
        num_array.append(int(n))
    testing=np.array(num_array)
    testin_matrix_C1=testing - np.array(C1_feature_var)
    testin_matrix_C2=testing - np.array(C2_feature_var)
    an=np.array(testin_matrix_C1)
    am=np.array(testin_matrix_C2)
    testin_matrix_C1_transpose=np.array(an.reshape(-1,1).tolist()).transpose()
    testin_matrix_C2_transpose=np.array(am.reshape(-1,1).tolist()).transpose()
    testin_matrix_C1=np.array(an.reshape(-1,1).tolist())
    testin_matrix_C2=np.array(am.reshape(-1,1).tolist())
    #function call
    navies(listC1,listC2,testin_matrix_C1_transpose,testin_matrix_C2_transpose,testin_matrix_C1,testin_matrix_C2)
    
    
    
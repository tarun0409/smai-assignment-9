import numpy as np
import pandas as pd
import random
import math
import time
import pickle

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

np.random.seed(1299827)

class Autoencoder:
    L = 3
    data_size = None
    W = None
    input_activation = 'linear'
    output_activation = 'linear'
    layer_activation = 'sigmoid'
    bottleneck_layer = 1
    
    def __init__(self,orig_data_size,compressed_data_sizes=[1],input_activation = 'linear',output_activation = 'linear',layer_activation = 'sigmoid'):
        self.L = len(compressed_data_sizes) + 2
        self.data_size = orig_data_size
        self.W = list()
        self.input_activation = input_activation
        self.output_activation = output_activation
        self.layer_activation = layer_activation
        n = [orig_data_size]+compressed_data_sizes+[orig_data_size]
        for i in range(0,self.L-1):
            self.W.append(np.random.rand(n[i],n[i+1]))
        self.bottleneck_layer = n.index(min(n))
            
    def g(self,z,activation):
        if activation == 'linear':
            return z
        if activation == 'sigmoid':
            z = np.clip(z,-709,36)
            return 1.0/(1.0+np.exp(-z))
        return z
    
    def g_prime(self,z,activation):
        if activation == 'linear':
            return np.ones(z.shape)
        if activation == 'sigmoid':
            g = self.g(z,activation)
            return np.multiply(g,(1.0-g))
    
    def compute_loss(self,orig_X):
        m = len(orig_X)
        a = orig_X
        h = self.g(a,self.input_activation)
        for i in range(0,self.L-2):
            a = np.dot(h,self.W[i])
            h = self.g(a,self.layer_activation)
        a_out = np.dot(h,self.W[len(self.W)-1])
        X_hat = self.g(a_out,self.output_activation)
        X = np.subtract(X_hat, orig_X)
        return (1.0/m)*np.sum(np.dot(X.T,X))
    
    def get_compressed_data(self,X):
        a = X
        h = self.g(a,self.input_activation)
        for i in range(0,self.bottleneck_layer):
            a = np.dot(h,self.W[i])
            h = self.g(a,self.layer_activation)
        return h
            
    def propagate(self, X):
        m = len(X)
        a = list()
        h = list()
        a.append(X)
        h.append(self.g(a[0],self.input_activation))
        for i in range(0,self.L-2):
            a.append(np.dot(h[i],self.W[i]))
            h.append(self.g(a[i+1],self.layer_activation))

        a_out = np.dot(h[len(h)-1],self.W[len(self.W)-1])
        X_hat = self.g(a_out,self.output_activation)
        
        dLdO = (2.0/m)*np.subtract(X_hat,X)
        dLda = np.multiply(dLdO,self.g_prime(a_out,self.output_activation))
        dW = list()
        for i in range(self.L-2,0,-1):
            dLdW = np.dot(h[i].T,dLda)
            dW = [dLdW]+dW 
            dLdh = np.dot(dLda,self.W[i].T)
            dLda = np.multiply(dLdh,self.g_prime(a[i],self.layer_activation))
        dLdW = np.dot(h[0].T,dLda)
        dW = [dLdW]+dW
        return dW
    
    def fit(self, X, alpha, epochs):
        for e in range(0,epochs):
            dW = self.propagate(X)
            for j in range(0,len(dW)):
                self.W[j] = self.W[j]-alpha*dW[j]
            curr_loss = self.compute_loss(X)
            print 'Cost after '+str(e+1)+' epochs: '+str(curr_loss)
    
    def save_weights(self):
        w_pickle = open('W_'+str(int(time.time())), 'wb')
        pickle.dump(self.W,w_pickle)
        w_pickle.close()
    
    def set_weights(self,file_name):
        w_pickle = open(file_name,'rb')
        self.W = pickle.load(w_pickle)
        w_pickle.close()


data = pd.read_csv("intrusion_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data[['duration', 'service', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']],
    data[['xAttack']],
    test_size=0.2,
    random_state=0)


for col in X_train:
    mean = X_train[col].mean()
    std = X_train[col].std()
    X_train[col] = (X_train[col] - mean)/std
    X_test[col] = (X_test[col]-mean)/std



ae = Autoencoder(len(X_train.columns),[14],'linear','linear','sigmoid')
orig_X = X_train.values



# ae.fit(orig_X,1e-2,50000)
# ae.save_weights()



ae.set_weights('W_1553848799.pickle')


test_X = X_test.values


compressed_X = ae.get_compressed_data(orig_X)
compressed_X = np.append(compressed_X,y_train,axis=1)
compressed_test_X = ae.get_compressed_data(test_X)
compressed_test_X = np.append(compressed_test_X,y_test,axis=1)
compressed_data = np.append(compressed_X,compressed_test_X,axis=0)
np.savetxt("compressed_intrusion_data_a.csv", compressed_data, delimiter=",",fmt="%s")



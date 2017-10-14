#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author Jesus Chavez

"""
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
class api:
  W = [];
  B = [];
  x_train = [];
  y_train=[];
  train_class=[];
  x_test=[];
  y_test=[];
  test_class=[]; 
  def __init__(self):
    np.set_printoptions(threshold=np.inf) 
    self.W, self.B = self.read_parameters();
    self.x_train, self.y_train, self.train_class, self.x_test, self.y_test, self.test_class = self.read_data(); 
  def relu(self,X):
      return np.maximum(X, 0)

  def sigmoid(self,X):
      return expit(X)

  def onehot(self,X):
      T = np.zeros((X.shape[0],np.max(X)+1))
      T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
      return T

  def feedForward(self,X,W,B):
      h0 = self.relu(np.dot(X, W[0]) + B[0])
      h1 = self.relu(np.dot(h0, W[1]) + B[1])
      out = self.sigmoid(np.dot(h1, W[2]) + B[2])
      return out

  def eval_out(self,out, test_class,y_test):
      pred = np.argmax(out, axis=1)
      pred = np.squeeze(np.asarray(pred))
      x = np.equal(pred,test_class)
      acc= float(sum(x))/float(len(x));
      loss = np.mean(np.square(y_test-out))
      return acc, pred, loss

  def confusion_matrix(self,Actual,Pred):
      cm=np.zeros((np.max(Actual)+1,np.max(Actual)+1), dtype=np.int)
      for i in range(len(Actual)):
          cm[Actual[i],Pred[i]]+=1
      return cm

  def read_data(self):
      x_train = np.loadtxt("xtrain.txt", delimiter=",")
      train_class = np.loadtxt("ytrain.txt", delimiter=",").astype(int)
      x_test = np.loadtxt("xtest.txt", delimiter=",")
      test_class = np.loadtxt("ytest.txt", delimiter=",").astype(int)
      x_train /= 255
      y_train = self.onehot(train_class)
      x_test /= 255
      y_test = self.onehot(test_class)
      return x_train, y_train, train_class, x_test, y_test, test_class

  def read_parameters(self):
      W=[]
      W.append(np.loadtxt("W0.txt", delimiter=","))
      W.append(np.loadtxt("W1.txt", delimiter=","))
      W.append(np.loadtxt("W2.txt", delimiter=","))
      B=[]
      B.append(np.loadtxt("B0.txt", delimiter=","))
      B.append(np.loadtxt("B1.txt", delimiter=","))
      B.append(np.loadtxt("B2.txt", delimiter=","))
      return W,B

  def display_errors(self,x_test,test_class,pred,n):
      error = np.flatnonzero(test_class-pred)    
      p = np.random.randint(len(error), size=np.minimum(n,len(error)))
      error = error[p]
      for j in range(n):
          i = error[j]
          plt.imshow((255*x_test[i]).reshape((28, 28)).astype(int), cmap=plt.get_cmap('gray'))
          title = 'True label:'+str(test_class[i])+'  Predicted label:'+str(pred[i])
          plt.title(title, fontsize=18, color='black')
          plt.show();
      




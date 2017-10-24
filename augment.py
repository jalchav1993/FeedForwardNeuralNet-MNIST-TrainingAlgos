#will recieve a training set x, and y
#create batch of n x -> y
#filtear each x with different vectors
#up up, up right, up left, right right, down right, down down, down left, left left.
#in this scrip i is row, j is col
#author: Jesus Chavez
import sys
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np;
from math import sqrt 
sys.path.insert(0,'./')
from ffneuralnet import *
class augment_model:
  DIRECTION = ['up-up', 'up-rt','up-lt', 'rt-rt', 'dn-rt','dn-dn', 'dn-lt', 'lt-lt'];
  def __init__(self, padding, vector_set_x, vector_set_y, shift):
    global IMG_SIZE;
    global STEP;
    global SHIFT;
    global PADDING;
    global LPADDING;
    global NPADDING;
    global OFFSET;
    global INSET;
    global XSET;
    global YSET;
    global N;
    global SET_SIZE;
    self.XSET = vector_set_x;
    self.YSET = vector_set_y;
    self.SET_SIZE = len(vector_set_x);
    self.IMG_SIZE = len(vector_set_x[0]);
    self.N = int(sqrt(self.IMG_SIZE));
    self.PPADDING = padding;
    self.SHIFT = shift;
    self.STEP = self.PPADDING -  self.SHIFT;
    self.LPADDING = 2 * self.PPADDING + self.N;
    self.NPADDING = self.PPADDING + self.N;
    self.alpha = (self.a0,self.a1,self.a2,self.a3) = (0, self.NPADDING, 0, self.NPADDING)
    self.beta = (self.b0,self.b1,self.b2,self.b3) = (0, self.NPADDING, self.PPADDING, self.LPADDING);
    self.gama = (self.g0,self.g1,self.g2,self.g3) = (self.PPADDING,self.LPADDING, 0, self.NPADDING);
    self.delta = (self.d0,self.d1,self.d2,self.d3) = (self.PPADDING, self.LPADDING, self.PPADDING, self.LPADDING);
    #find out this
    #only works when shift is 1, must add offset
    self.OFFSET ={
      'up-up':self.alpha, 
      'up-rt':self.beta,
      'up-lt':self.alpha, 
      'rt-rt':self.beta, 
      'dn-rt':self.delta,
      'dn-dn':self.gama, 
      'dn-lt':self.gama, 
      'lt-lt':self.alpha
    }
    self.INSET = {
      'up-up':self.gama, 
      'up-rt':self.gama,
      'up-lt':self.delta, 
      'rt-rt':self.alpha, 
      'dn-rt':self.alpha,
      'dn-dn':self.alpha, 
      'dn-lt':self.beta, 
      'lt-lt':self.beta
    }

  def shift_vect(self, vector, direction):
    (i0,i1,i2,i3) = self.INSET[direction];
    (o0,o1,o2,o3) = self.OFFSET[direction];
    new_vector = np.array(np.zeros((self.IMG_SIZE,), dtype=np.float));
    new_vector = self.square_vect(new_vector);
    new_vector = self.pad_vect(new_vector);
    new_vector[o0:o1, o2:o3] = vector[i0:i1, i2:i3]
    return new_vector;
  
  def square_vect(self, vector):
    #sqrt(edge) must be even
    edge = len(vector);
    diagonal = int(sqrt(edge));
    return np.reshape(vector, (-1, diagonal));
  
  def pad_vect(self, vector):
    return np.lib.pad(vector,self.PPADDING,self.pad_zero);
  
  def pad_zero(self, vector, pad_width, iaxis, kwargs):
    # used this solution https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.pad.html
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector
  def build (self):
    c_x = [];
    c_y = [];
    for i in range(self.SET_SIZE):
      #8 vectors
      for d in self.DIRECTION:
        c_x.append(self.filterimg(self.XSET[i], d));
        c_y.append(self.YSET[i]);
    return (c_x, c_y);

  def filterimg(self, vector, direction):
    # 1 make vector square
    # 2 pad this vector
    # 3 shift inner numbers
    # 4 crop the centerpiece
    vector = self.square_vect(vector);
    vector = self.pad_vect(vector);
    vector = self.shift_vect(vector, direction);
    vector = self.trim(vector);
    return vector.flatten();

  def trim(self, vector):
    return vector[self.PPADDING:self.NPADDING, self.PPADDING:self.NPADDING];


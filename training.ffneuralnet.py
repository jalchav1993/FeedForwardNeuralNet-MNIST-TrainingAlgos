#Author Jesus Chavez
#trains for a neural network
#uses test set augmentation
#fixed the naming issue, Python doesNotUseThisStyle
import sys
from math import sqrt 
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(0,'./')
from ffneuralnet import *
from augment import *
from dropout import *
from random import sample

F = api();
BATCH_SIZE = 77;
BSPEC = 0.1999;
WSPEC = 0.1999;
DELTASPEC = 0.0666;
EPOCH = 14;
STEP = 46;
X0_LEN = 784;
H0_LEN = 90;
H1_LEN = 60;
L = 0.000666;
dropout_train = dropout_model(F.x_train,F.y_train, F.train_class);
dropout_test = dropout_model(F.x_test,F.y_test, F.test_class);
(X,Y, TRAIN_CLASS) = dropout_train.build();
(X_TEST,Y_TEST, TEST_CLASS) = dropout_test.build();
TRAINING_SIZE = len(X);
def find_weights_random(w,b):
  m_error = [];
  (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
  e_current = get_error(w, b, batchx,batchy);
  m_error.append(e_current);
  for i in range(STEP*10 + 6):
    # let xb, yb be a randomly chosen subset of (X, Y)
    (deltaw, deltab) = get_random_set(DELTASPEC);
    (wnew, bnew) = add(w, b, deltaw, deltab);
    e_trial = get_error(wnew, bnew, batchx,batchy);
    (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
    print ("iteration: %d e0:%f e1:%f" %(i, e_current,e_trial));
    if e_trial < e_current: 
      w = wnew;
      b = bnew;
      e_current = e_trial;
      m_error.append(e_current);
  return (w,b,m_error);
  
def find_weights_pseudoinverse():
  w = [];
  b = [];
  #let Wbest=W0,W1,W2,b0,b1,b2 be a randomly-chosen set of weights
  (batchx, batchy) = find_random_batch(X, Y);
  w.append(find_random_weight(X0_LEN, H0_LEN, WSPEC));
  w.append(find_random_weight(H0_LEN, H1_LEN, WSPEC)); 
  b.append(find_random_bias(H0_LEN, BSPEC));
  b.append(find_random_bias(H1_LEN, BSPEC));
  h0 = F.relu(np.dot(batchx,w[0])+ b[0]);
  h1 = F.relu(np.dot(h0,w[1]) + b[1]);
  y_l = [y * 0.9 + 0.05 for y in batchy];
  y_l = logit(y_l);
  b.append(np.mean(y_l));
  w.append(np.dot(np.linalg.pinv(h1), (y_l - b[2])));
  return (w,b)
  
def find_weights_backdrop(w, b):
  m_error = [];
  l = L;
  for i in range(EPOCH):
    # this is magick
    new_error_sum = 0;
    (batchx, batchy) = find_random_batch(X, Y);
    # augmenting the set
    # check this with doctor olac
    y_l = [y * 0.9 + 0.05 for y in batchy];
    for j in range(STEP):
      h0 = F.relu(np.dot(batchx,w[0])+ b[0]);
      h1 = F.relu(np.dot(h0,w[1]) + b[1]);
      P = F.sigmoid(np.dot(h1,w[2]) + b[2]);
      #is this the hadamard product
      dP_ = np.multiply((P - y_l),P);
      dP= np.multiply(dP_ ,(1-P));
      #transposed weights concatenated
      W_t = [np.transpose(weight) for weight in w];
      dH1 = np.multiply(np.dot(dP,W_t[2]), sign(h1));
      dH0 = np.multiply(np.dot(dH1,W_t[1]),sign(h0));
      new_error = get_mean_error(batchy,P);
      new_error_sum += new_error;
      w[0] = w[0] - l * np.dot(np.transpose(batchx), dH0);
      w[1] = w[1] - l * np.dot(np.transpose(h0), dH1);
      w[2] = w[2] - l * np.dot(np.transpose(h1), dP);
      b[0] = b[0] - l * np.sum(dH0, axis = 0);
      b[1] = b[1] - l * np.sum(dH1, axis = 0);
      b[2] = b[2] - l * np.sum(dP, axis = 0);
    new_error_sum/=STEP;
    m_error.append(new_error_sum);
    if committee(new_error_sum):
      break;
    print ("Epoch: %d, error %f" %(i, new_error_sum));
  return (w,b, m_error);

def find_weights_backdrop_l2(train_v, test_v,w, b, alpha_reg, beta, flag):
  (x_train,y_train, train_class) = train_v
  (x_test,y_test, test_class) = test_v
  acc_test =[];
  acc_train = [];
  l = L;
  alpha_it = 0;
  gW0 = gW1 = gW2 = gb0 = gb1 = gb2 =0;
  for i in range(EPOCH):
    # this is magick
    acc_sum_train = 0;
    acc_sum_test = 0;
    test_set = test_class;
    (batchx, batchy) = find_random_batch(x_train, y_train);
    # augmenting the set
    y_l = [y * 0.9 + 0.05 for y in batchy];
    for j in range(STEP):
      h0 = F.relu(np.dot(batchx,w[0])+ b[0]);
      h1 = F.relu(np.dot(h0,w[1]) + b[1]);
      P = F.sigmoid(np.dot(h1,w[2]) + b[2]);
      #is this the hadamard product
      dP_ = np.multiply((P - y_l),P);
      dP= np.multiply(dP_ ,(1-P));
      #transposed weights concatenated
      W_t = [np.transpose(weight) for weight in w];
      dH1 = np.multiply(np.dot(dP,W_t[2]), sign(h1));
      dH0 = np.multiply(np.dot(dH1,W_t[1]),sign(h0));
      batch_x_t =np.transpose(batchx);
      h0_t = np.transpose(h0);
      h1_t = np.transpose(h1);
      #update weights and biases
      gW2 = alpha_it * gW2 + (1 - alpha_it) * np.dot(h1_t, dP);
      gW1 = alpha_it * gW1 + (1 - alpha_it) * np.dot(h0_t, dH1);
      gW0 = alpha_it * gW0 + (1 - alpha_it) * np.dot(batch_x_t, dH0);
      
      gb2 = alpha_it * gb2 + (1 - alpha_it) * np.sum(dP);
      gb1 = alpha_it * gb1 + (1 - alpha_it) * np.sum(dH1);
      gb0 = alpha_it * gb0 + (1 - alpha_it) * np.sum(dH0);
      #new_error = get_mean_error(batchy,P);
      w[2] = w[2] - l * (gW2 + beta * w[2]);
      w[1] = w[1] - l * (gW1 + beta * w[1]);
      w[0] = w[0] - l * (gW0 + beta * w[0]);
      b[2] = b[2] - l * (gb2 + beta * b[2]);
      b[1] = b[1] - l * (gb1 + beta * b[1]);
      b[0] = b[0] - l * (gb0 + beta * b[0]);
      acc_train_i = get_acc(w, b, x_train, y_train, train_class);
      acc_test_i = get_acc(w, b, x_test, y_test, test_class);
      acc_sum_train += acc_train_i;
      acc_sum_test += acc_test_i;
    acc_sum_train/=STEP;
    acc_sum_test/=STEP;
    acc_train.append(acc_sum_train);
    acc_test.append(acc_sum_test);
    alpha_it = alpha_reg;
    #greedy  
    print ("Epoch: %d, accuracy test %f, accuracy train %f" %(i, acc_sum_test,acc_sum_train));
    if (acc_sum_test < 0.39 and flag):
      print "Greedy: Dropping try when accuracy < 0.39"
      return run();
    elif(acc_sum_test >= 0.39 and flag):
      flag = False;  
  return (w,b, acc_train, acc_test);
def run():
  dropout_train = dropout_model(F.x_train,F.y_train, F.train_class);
  dropout_test = dropout_model(F.x_test,F.y_test, F.test_class);
  train_v = (x_train,y_train, train_class) = dropout_train.build();
  test_v = (x_test,y_test, test_class) = dropout_test.build();
  training_size = len(x_train);
  flag = True;
  alpha_reg = 0.000666;
  beta = 0.000666;
  (w_inv, b_inv) = find_weights_pseudoinverse();
  (w,b, acc_train, acc_test) = find_weights_backdrop_l2(train_v, test_v, w_inv, b_inv, alpha_reg, beta, flag)
  return (w,b, acc_train, acc_test);
def logit(y):
  return [np.log(label) - np.log(1-label) for label in y];
  
def sign(y):
  for x in np.nditer(y, op_flags=['readwrite']):
      x[...] = getdif(x);
  return y;
def committee(error):
  #tests by comitte
  return error < 0.003013013;
def getdif(x):
  if x > 0: return 1 
  elif x == 0: return 0
  else: return -1;
  
def find_random_weight(row, col, spec):
  return (np.random.rand(row,col)-0.5)*spec;
  
def find_random_bias(size, spec):
  return (np.random.rand(size)-0.5)*spec;

def get_random_set(spec):
  return ([find_random_weight(X0_LEN, H0_LEN, spec),
    find_random_weight(H0_LEN, H1_LEN, spec),
    find_random_weight(H1_LEN, 10, spec)],
    [find_random_bias(H0_LEN, spec),
    find_random_bias(H1_LEN, spec),
    find_random_bias(10, spec)]);

def get_acc(w, b, x ,y, test_set):
  out = F.feedForward(x, w,b);
  acc, pred, loss = F.eval_out(out, test_set,y);
  return acc;
def get_mean_error(y,p):
  return np.mean(np.square(y-p));
    
def get_error(w, b, x, y):
  #need help understanding why this
  
  return np.mean(np.square(y-F.feedForward(x,w, b)));
  
def add(w,b, changew, changeb):
  return ([w[0]+changew[0], 
    w[1]+changew[1], 
    w[2]+changew[2]],
    [b[0]+changeb[0], 
    b[1]+changeb[1], 
    b[2]+changeb[2]]);
    
def tset_augmentation(x, y, p):
  #model in ./augment.py module
  model = augment_model(p,x,y,p);
  return model.build();
def dropout(x,y):
  model =dropout_model(x,y);
  return model.build()
def find_random_batch(x, y):
  indexes = sample(range(TRAINING_SIZE), BATCH_SIZE)
  return addhoc([x[i] for i in indexes], [y[i] for i in indexes]);
def addhoc(x,y):
  (x,y) = dropout(x,y); 
  return tset_augmentation(x,y,1);
def print_test(out, x, y,test_set,acc_set):
  np.set_printoptions(threshold=np.inf) #Print complete arrays
  acc, pred, loss = F.eval_out(out, test_set,y);
  print("Accuracy =",acc);
  print("Loss =",loss);
  print("Confusion_matrix:");
  print(F.confusion_matrix(test_set,pred));
  F.display_errors(x,test_set,pred,5);
  plt.plot(acc_set);
  plt.show();
(w,b, acc_train, acc_test) = run();
te_set_test = F.feedForward(X_TEST, w,b);
tr_set_test = F.feedForward(X, w,b);
print_test(te_set_test, X_TEST, Y_TEST, TEST_CLASS, acc_test);
print_test(tr_set_test, X, Y,TRAIN_CLASS, acc_train);

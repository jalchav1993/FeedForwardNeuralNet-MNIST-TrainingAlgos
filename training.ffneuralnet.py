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
F = api();
TRAINING_SIZE = len(F.x_train);
BATCH_SIZE = 77;
BSPEC = 0.1999;
WSPEC = 0.1999;
DELTASPEC = 0.0666;
EPOCH = 30;
STEP = 66;
X0_LEN = 784;
H0_LEN = 60;
H1_LEN = 30;
L = 0.00066969;

def find_weights_random(w,b):
  m_error = [];
  (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
  e_current = get_error(w, b, batchx,batchy);
  m_error.append(e_current);
  for i in range(STEP*10 + 6):
#     # let xb, yb be a randomly chosen subset of (X, Y)
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
  (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
  print X0_LEN;
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
    (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
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
def find_weights_backdrop_l2(w, b, alpha_reg, beta_reg):
  m_error = [];
  l = L;
  alpha_it = 0;
  for i in range(EPOCH):
    # this is magick
    new_error_sum = 0;
    (batchx, batchy) = find_random_batch(F.x_train, F.y_train);
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
def update_lambda(l, new_error, old_error):
  #check this
  if (new_error > old_error):
    operator = -1;
  else:
    operator = 1;
  error_convergence = np.absolute(new_error - old_error);
  print "error %f"% error_convergence;
  if(error_convergence>= 0.15): 
    return l + operator*0.000000701;
  elif(error_convergence>= 0.04 and error_convergence < 0.15):
    return l + operator*0.00001050;
  elif(error_convergence>= 0.01 and error_convergence < 0.04):
    return l + operator*0.00001100;
  elif(error_convergence>= 0.002 and error_convergence < 0.01): 
    return l + operator*0.00002125;
  else:
    return l + 0.0004150;
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
    
def tset_augmentation(X, Y):
  #model in ./augment.py module
  model = augment_model(1,X,Y,1); 
  return model.build();

def find_random_batch(x, y):
  indexes = np.random.randint(0,TRAINING_SIZE,BATCH_SIZE);
  return augment([x[i] for i in indexes], [y[i] for i in indexes]);
def augment(x,y):
  #addhoc 
  return tset_augmentation(x,y);
def print_test(out, m_error):
  np.set_printoptions(threshold=np.inf) #Print complete arrays
  acc, pred, loss = F.eval_out(out, F.test_class,F.y_test);
  print("Accuracy =",acc);
  print("Loss =",loss);
  print("Confusion_matrix:");
  print(F.confusion_matrix(F.test_class,pred));
  F.display_errors(F.x_test,F.test_class,pred,5);
  plt.plot(m_error);
  plt.show();
#(w_init, b_init) = get_random_set(WSPEC);
#algorithm 1
#(w_rand, b_rand, m_error_rand) = find_weights_random(w_init,b_init);
# #algo 2
(w_psinv, b_psinv) = find_weights_pseudoinverse();
# #algo 3
#(w_bd, b_bd, m_error_bd) = find_weights_backdrop(w_init,b_init);
# #algo 4
(w_bd_inv, b_bd_inv, m_error_inv) = find_weights_backdrop(w_psinv,b_psinv);
# #printing
#print_test(F.feedForward(F.x_test, w_rand,b_rand), m_error_rand);
#print_test(F.feedForward(F.x_test, w_psinv,b_psinv),'');
#print_test(F.feedForward(F.x_test, w_bd,b_bd),m_error_bd
print_test(F.feedForward(F.x_test, w_bd_inv,b_bd_inv),m_error_inv);

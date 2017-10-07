
import sys
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(0,'./')
from ffneuralnet import *
F = api();
TRAINING_SIZE = len(F.x_train);
BATCH_SIZE = 777;
BSPEC = 0.1999;
WSPEC = 0.1999;
DELTASPEC = 0.0666;
EPOCH = 99;
STEP = 66;
H0_LEN = 60;
H1_LEN = 30;
L = 0.000969;
def randomSet():
  return ();
def findWeightsRandom(w,b):
  m_error = [];
  (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
  e_current = get_error(w, b, batchx,batchy);
  m_error.append(e_current);
  for i in range(STEP*10 + 6):
#     # let xb, yb be a randomly chosen subset of (X, Y)
    (deltaw, deltab) = getRandomSet(DELTASPEC);
    (wnew, bnew) = add(w, b, deltaw, deltab);
    e_trial = get_error(wnew, bnew, batchx,batchy);
    (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
    print ("iteration: %d e0:%f e1:%f" %(i, e_current,e_trial));
    if e_trial < e_current: 
      w = wnew;
      b = bnew;
      e_current = e_trial;
      m_error.append(e_current);
  return (w,b,m_error);
def findWeightsPseudoinverse():
  w = [];
  b = [];
  #let Wbest=W0,W1,W2,b0,b1,b2 be a randomly-chosen set of weights
  w.append(findRandomWeight(784, H0_LEN, WSPEC));
  w.append(findRandomWeight(H0_LEN, H1_LEN, WSPEC)); 
  b.append(findRandomBias(H0_LEN, BSPEC));
  b.append(findRandomBias(H1_LEN, BSPEC));
  (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
  h0 = F.relu(np.dot(batchx,w[0])+ b[0]);
  h1 = F.relu(np.dot(h0,w[1]) + b[1]);
  y_l = [y * 0.9 + 0.05 for y in batchy];
  y_l = logit(y_l);
  b.append(np.mean(y_l));
  w.append(np.dot(np.linalg.pinv(h1), (y_l - b[2])));
  return (w,b)
def findWeightsBackDrop(w, b):
  m_error = [];
  for i in range(EPOCH):
    #this is magick
    new_error_sum = 0;
    (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
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
      w[0] = w[0] - L * np.dot(np.transpose(batchx), dH0);
      w[1] = w[1] - L * np.dot(np.transpose(h0), dH1);
      w[2] = w[2] - L * np.dot(np.transpose(h1), dP);
      b[0] = b[0] - L * np.sum(dH0, axis = 0);
      b[1] = b[1] - L * np.sum(dH1, axis = 0);
      b[2] = b[2] - L * np.sum(dP, axis = 0);
      #print new_error;
      #print (i, j);
    new_error_sum/=STEP;
    m_error.append(new_error_sum);
    if new_error_sum < 0.00444:
      break;
    print ("Epoch: %d, error %f" %(i, new_error_sum));
  return (w,b, m_error);
def logit(y):
  return [np.log(label) - np.log(1-label) for label in y];
  
def sign(y):
  for x in np.nditer(y, op_flags=['readwrite']):
      x[...] = getdif(x);
  return y;
  
def getdif(x):
  if x > 0: return 1 
  elif x == 0: return 0
  else: return -1;
  
def findRandomWeight(row, col, spec):
  return (np.random.rand(row,col)-0.5)*spec;
  
def findRandomBias(size, spec):
  return (np.random.rand(size)-0.5)*spec;

def getRandomSet(spec):
  return ([findRandomWeight(784, H0_LEN, spec),
    findRandomWeight(H0_LEN, H1_LEN, spec),
    findRandomWeight(H1_LEN, 10, spec)],
    [findRandomBias(H0_LEN, spec),
    findRandomBias(H1_LEN, spec),
    findRandomBias(10, spec)]);
def getlambda(n):
  return [0.00066 for i in range(n)];
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
    
def findRandomBatch(x, y):
  indexes = np.random.randint(0,TRAINING_SIZE,BATCH_SIZE);
  return ([x[i] for i in indexes], [y[i] for i in indexes]);
  
def printTest(out, m_error):
  np.set_printoptions(threshold=np.inf) #Print complete arrays
  acc, pred, loss = F.eval_out(out, F.test_class,F.y_test);
  print("Accuracy =",acc);
  print("Loss =",loss);
  print("Confusion_matrix:");
  print(F.confusion_matrix(F.test_class,pred));
  F.display_errors(F.x_test,F.test_class,pred,5);
  plt.plot(m_error);
  plt.show();
#(w_init, b_init) = getRandomSet(WSPEC);
#algorithm 1
#(w_rand, b_rand, m_error_rand) = findWeightsRandom(w_init,b_init);
# #algo 2
(w_psinv, b_psinv) = findWeightsPseudoinverse();
# #algo 3
#(w_bd, b_bd, m_error_bd) = findWeightsBackDrop(w_init,b_init);
# #algo 4
(w_bd_inv, b_bd_inv, m_error_inv) = findWeightsBackDrop(w_psinv,b_psinv);
# #printing
#printTest(F.feedForward(F.x_test, w_rand,b_rand), m_error_rand);
#printTest(F.feedForward(F.x_test, w_psinv,b_psinv),'');
#printTest(F.feedForward(F.x_test, w_bd,b_bd),m_error_bd);
printTest(F.feedForward(F.x_test, w_bd_inv,b_bd_inv),m_error_inv);

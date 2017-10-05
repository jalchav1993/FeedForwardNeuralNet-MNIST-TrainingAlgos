
import sys
from scipy.special import expit
import numpy as np
sys.path.insert(0,'./')
from ffneuralnet import *
F = api();
TRAINING_SIZE = len(F.x_train);
BATCH_SIZE = 777;
BSPEC = 0.1999;
WSPEC = 0.1999;
DELTASPEC = 0.0666;
STEP = 1313;
H0_LEN = 90;
H1_LEN = 60;
L = 0.001;
def findWeightsRandom():
  w = [];
  b = [];
  #let Wbest=W0,W1,W2,b0,b1,b2 be a randomly-chosen set of weights
  w.append(findRandomWeight(784, H0_LEN, WSPEC));
  w.append(findRandomWeight(H0_LEN, H1_LEN, WSPEC)); 
  w.append(findRandomWeight(H1_LEN, 10, WSPEC));
  b.append(findRandomBias(H0_LEN, BSPEC));
  b.append(findRandomBias(H1_LEN, BSPEC));
  b.append(findRandomBias(10, BSPEC));
  (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
  e_current = get_error(w, b, batchx,batchy);
  for i in range(STEP):
#     # let xb, yb be a randomly chosen subset of (X, Y)
    (deltaw, deltab) = getDelta(w, b);
    (wnew, bnew) = add(w, b, deltaw, deltab);
    e_trial = get_error(wnew, bnew, batchx,batchy);
    (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
    print ("iteration: %d e0:%f e1:%f" %(i, e_current,e_trial));
    if e_trial < e_current: 
      w = wnew;
      b = bnew;
      e_current = e_trial;
  return (w,b);
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
def findWeightsBackDrop():
  w = [];
  b = [];
  #let Wbest=W0,W1,W2,b0,b1,b2 be a randomly-chosen set of weights
  w.append(findRandomWeight(784, H0_LEN, WSPEC));
  w.append(findRandomWeight(H0_LEN, H1_LEN, WSPEC)); 
  w.append(findRandomWeight(H1_LEN, 10, WSPEC));
  b.append(findRandomBias(H0_LEN, BSPEC));
  b.append(findRandomBias(H1_LEN, BSPEC));
  b.append(findRandomBias(10, BSPEC));
  (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
  for i in range(100):
    print i;
    #magick
    (batchx, batchy) = findRandomBatch(F.x_train, F.y_train);
    h0 = F.relu(np.dot(batchx,w[0])+ b[0]);
    h1 = F.relu(np.dot(h0,w[1]) + b[1]);
    P = F.sigmoid(np.dot(h1,w[2]) + b[2]);
    #dP = np.multiply(np.multiply((P - batchy), P), (1-P));
    dP_ = np.multiply((P - batchy), P);
    dP= np.multiply(dP_, (1-P));
    #transposed weights concatenated
    W_t = [weight.T for weight in w];
    dH1 = np.multiply(np.dot(dP,W_t[2]), sign(h1));
    dH0 = np.multiply(np.dot(dH1,W_t[1]), sign(h0));
    #what is the significance of this transformation? I wonder
    w[2] = w[2] - L * np.dot(h1.T, dP);
    w[1] = w[1] - L * np.dot(h0.T, dH1);
    w[0] = w[0] - L * np.dot(np.array(batchx).T, dH0);
    b[2] = b[2] - L * np.sum(dP, axis = 0);
    b[1] = b[1] - L * np.sum(dH1, axis = 0);
    b[0] = b[0] - L * np.sum(dH0, axis = 0);
  return (w,b);
def logit(y):
  return [np.log(label) - np.log(1-label) for label in y];
  
def sign(y):
  for x in np.nditer(y, op_flags=['readwrite']):
      x[...] = getdif(x);
  return y;
  
def getdif(x):
  if x > 1: return 1 
  elif x == 0: return 0
  else: return -1;
  
def findRandomWeight(row, col, spec):
  return (np.random.rand(row,col)-0.5)*spec;
  
def findRandomBias(size, spec):
  return (np.random.rand(size)-0.5)*spec;

def getDelta(w, b):
  return ([findRandomWeight(784, H0_LEN, DELTASPEC),
    findRandomWeight(H0_LEN, H1_LEN, DELTASPEC),
    findRandomWeight(H1_LEN, 10, DELTASPEC)],
    [findRandomBias(H0_LEN, DELTASPEC),
    findRandomBias(H1_LEN, DELTASPEC),
    findRandomBias(10, DELTASPEC)]);
    
def get_error(w, b, x, y):
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

#(w, b) = findWeightsPseudoinverse();
(w, b) = findWeightsBackDrop();
out = F.feedForward(F.x_test, w, b);
np.set_printoptions(threshold=np.inf) #Print complete arrays
acc, pred, loss = F.eval_out(out, F.test_class,F.y_test);
print("Accuracy =",acc);
print("Loss =",loss);
print("Confusion_matrix:");
print(F.confusion_matrix(F.test_class,pred));
F.display_errors(F.x_test,F.test_class,pred,5);

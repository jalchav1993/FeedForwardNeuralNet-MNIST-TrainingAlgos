import sys
import numpy as np;
class dropout_model:
  def __init__(self, vector_set_x, vector_set_y, set_class = None):
    self.x = vector_set_x;
    self.y = vector_set_y;
    self.set_class = set_class;
  
  def build(self):
    lebel_count = self.get_label_count(self.y);
    if self.set_class is not None:
      (x,y, set_class) = self.dropout(lebel_count, self.x, self.y, self.set_class);
      #print "start shapes"
      #print np.shape(self.set_class);
      #print np.shape(set_class);
      #print "end shapes"
      return (x,y, set_class);
    (x,y) = self.dropout(lebel_count, self.x, self.y);
    return (x,y);
  
  def get_label_count (self, y):
    label_count = [0,0,0,0,0,0,0,0,0,0];
    for i in range(len(y)):
      y_label = np.argmax(self.y[i]);
      label_count[y_label]+=1;
    return label_count;
  
  def dropout(self, label_count,x ,y, set_class = None):
    avg = np.sum(label_count)/len(label_count);
    avg = int(avg + np.argmin(label_count))/2
    #print avg;
    #print label_count
    pop_list = [];
    for i in range(len(y)):
      label = np.argmax(y[i]);
      if label_count[label] > avg:
        pop_list.append(i);
        label_count[label]-=1;
    x = np.delete(x, pop_list, 0);
    y = np.delete(y, pop_list, 0);
    if set_class is not None:
      #print "break"
      #print np.shape(set_class);
      set_class = np.delete(set_class, pop_list, 0);
      return (x, y, set_class);
    return (x,y);
import numpy as np
import scipy as sp
import mnist
np.random.seed(3) # seed random number generator

class neural_net_feed_forward_gradient():
  # class attributes
  train_labels = mnist.train_labels()
  test_labels = mnist.test_labels()
  n_classes = 10 # number of classes (digits in this case)
  batch_size = 20 # batch size for training
  def __init__(self):
    # flatten images
    train_images = mnist.train_images()
    self.train_images = self.image_reshaper(train_images)
    test_images = mnist.test_images()
    self.test_images = self.image_reshaper(test_images)
    # derived parameters
    self.n_train_images = np.shape(self.train_images)[0]
    self.n_test_images = np.shape(self.test_images)[0]
    self.n_pixels = np.shape(self.train_images)[1]
    self.n_batches = int(np.floor(self.n_train_images/self.batch_size))
    # initialize
    self.W = 1e-3*( # weights, array #inputs x #classes
      np.random.random([self.n_pixels,self.n_classes]) - .5
    ) 
    self.B = 1e-3*( # biases, array 1 x #classes
      np.random.random([1,self.n_classes]) - .5
    ) 
  def train(self):
    for batch_number in range(0,self.n_batches):
      input_array = np.take(
        self.train_images,
        batch_number*self.batch_size+np.arange(0,self.batch_size),
        axis=0
      )
      loss = self.forward_pass(input_array,batch_number)
      # print("loss:\n")
      print(loss)
  def forward_pass(self,input_array,batch_number):
    logits = self.logitser(input_array) # batch_size x n_classes
    train_labels = np.take( # correct answers ... batch_size x 1
      self.train_labels,
      batch_number*self.batch_size+np.arange(0,self.batch_size)
    )
    logits_labeled = np.empty(self.batch_size) # initialize
    for i,train_label in enumerate(train_labels):
      logits_labeled[i] = logits[i,train_label] # correct logits ... batch_size x 1
    probabilities_labeled = self.probabilityer(logits_labeled,logits)
    return self.losser(probabilities_labeled)
  def backward_pass(self):
    return
  def logitser(self,input_array):
    # (1.25)
    # but input_array can be batch_size x #inputs
    # returns array size batch_size x n_classes
    return input_array @ self.W + self.B
  def probabilityer(self,logits_labeled,logits):
    # (1.26) ... just softmax
    numerator = np.exp(logits_labeled)
    denominator = np.sum(np.exp(logits),axis=1)
    return numerator/denominator
  def losser(self,probabilities_labeled):
    return -np.log(probabilities_labeled)
  def image_reshaper(self,img):
    return img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
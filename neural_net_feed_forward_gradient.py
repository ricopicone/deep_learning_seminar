import numpy as np
import scipy as sp
import mnist
np.random.seed(3) # seed random number generator

class neural_net_feed_forward_gradient():
  # class attributes
  train_answers = mnist.train_labels()
  test_answers = mnist.test_labels()
  n_classes = 10 # number of classes (digits in this case)
  batch_size = 20 # batch size for training
  learning_rate = 1e-4 # learning rate
  def __init__(self):
    # flatten images
    train_images = mnist.train_images()
    self.train_images = self.image_reshaper(train_images)/255
    test_images = mnist.test_images()
    self.test_images = self.image_reshaper(test_images)/255
    # derived parameters
    self.n_train_images = np.shape(self.train_images)[0]
    self.n_test_images = np.shape(self.test_images)[0]
    self.n_pixels = np.shape(self.train_images)[1]
    self.n_batches = int(np.floor(self.n_train_images/self.batch_size))
    self.n_batches_test = int(np.floor(self.n_test_images/self.batch_size))
    # initialize
    self.W = 1e-7*( # weights, array n_pixels x n_classes
      np.random.random([self.n_pixels,self.n_classes]) - .5
    ) 
    self.B = 1e-7*( # biases, array n_classes
      np.random.random([self.n_classes]) - .5
    ) 
  def randomize_images(self):
    assert len(a) == len(b)
    perm = numpy.random.permutation(n_train_images)
    self.train_answers = self.train_answers[perm]
    self.train_images = self.train_answers[perm]
  def train_once(self):
    for batch_number in range(0,self.n_batches):
      input_array = np.take( # batch_size x n_pixels
        self.train_images,
        batch_number*self.batch_size+np.arange(0,self.batch_size),
        axis=0
      )
      fp = self.forward_pass(input_array,batch_number)
      bp = self.backward_pass(input_array,batch_number,fp)
      # print(bp['delta_b'])
  def test(self):
    our_estimates = np.empty([self.n_test_images])
    for batch_number in range(0,self.n_batches_test):
      indices = batch_number*self.batch_size + \
        np.arange(0,self.batch_size)
      input_array = np.take( # batch_size x n_pixels
        self.train_images,
        indices,
        axis=0
      )
      fp = self.forward_pass(input_array,batch_number)
      our_estimates[indices] = fp['p_all'].argmax(axis=1)
    test_pass_fail_rate = self.test_pass_fail_rate(our_estimates)
    return test_pass_fail_rate
  def test_pass_fail_rate(self,our_estimates):
    test = 1*(our_estimates == self.test_answers) # 1* casts to int
    pass_fail_rate = test.sum()/test.size
    return pass_fail_rate
  def train(self):
    for i in range(0,10):
      # self.randomize_images() # not sure if we should
      self.train_once()
      test_pass_fail_rate = self.test()
      print(f'accuracy for pass {i:d}: {test_pass_fail_rate:.1%}')
  def forward_pass(self,input_array,batch_number,is_test=False):
    fp = dict()
    logits = self.logitser(input_array) # batch_size x n_classes
    fp['train_answers'] = np.take( # correct answers ... batch_size x 1
      self.train_answers,
      batch_number*self.batch_size+np.arange(0,self.batch_size)
    )
    logits_answers = np.empty(self.batch_size) # initialize
    for i,train_answer in enumerate(fp['train_answers']):
      logits_answers[i] = logits[i,train_answer] # correct logits ... batch_size x 1
    p = self.probabilityer(logits_answers,logits)
    fp['loss'] = self.losser(p['answers'])
    fp['p_answers'] = p['answers']
    fp['p_all'] = p['all']
    return fp
  def backward_pass(self,input_array,batch_number,fp):
    # return delta weights n_pixels x n_classes
    # return delta biases n_classes x 1
    bp = dict()
    dX_dl = np.empty([self.batch_size,self.n_classes])
    # (1.18)
    for i in range(0,self.batch_size): # each image
      for j in range(0,self.n_classes): # each class
        # print('p_all[i,j]')
        # print(fp['p_all'][i,j])
        if j == fp['train_answers'][i]: # j == a
          dX_dl[i,j] = -(1-fp['p_all'][i,j])
        else: # j != a
          dX_dl[i,j] = -fp['p_all'][i,j]
    bp['delta_w'] = -self.learning_rate*input_array.T@dX_dl # contracts batch_size axis
    # print(bp['delta_w'][25,:])
    bp['delta_b'] = -self.learning_rate*np.sum(dX_dl,axis=0) # sums batch_size axis
    # update weights and biases
    # print(np.shape(self.W))
    # print(np.shape(bp['delta_w']))
    # print(np.shape(self.B))
    # print(np.shape(bp['delta_b']))
    self.W = self.W+bp['delta_w']
    self.B = self.B+bp['delta_b']
    return bp
  def logitser(self,input_array):
    # (1.25)
    # but input_array can be batch_size x #inputs
    # returns array size batch_size x n_classes
    return input_array @ self.W + self.B
  def probabilityer(self,logits_answers,logits):
    # (1.26) ... just softmax
    numerator_answers = np.exp(logits_answers)
    numerator_all = np.exp(logits)
    denominator = np.sum(np.exp(logits),axis=1)
    # print('logits')
    # print(logits)
    p = dict()
    p['answers'] = numerator_answers/denominator
    p['all'] = (numerator_all.T/denominator).T # transposes for broadcasting
    return p
  def losser(self,probabilities_answers):
    return -np.log(probabilities_answers)
  def image_reshaper(self,img):
    return img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
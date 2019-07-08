import numpy as np
import scipy as sp
import mnist
from PIL import Image
from IPython.display import display
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
    perm = np.random.permutation(self.n_train_images)
    self.train_answers = self.train_answers[perm]
    self.train_images = self.train_images[perm]
  def test(self):
    our_estimates = np.empty([self.n_test_images]) # initialize
    for batch_number in range(0,self.n_batches_test):
      indices = batch_number*self.batch_size + \
        np.arange(0,self.batch_size)
      input_array = np.take( # batch_size x n_pixels
        self.test_images,
        indices,
        axis=0
      )
      fp = self.forward_pass(input_array,batch_number,is_test=True)
      our_estimates[indices] = fp['p_all'].argmax(axis=1)
    test_pass_fail_rate = self.test_pass_fail_rate(our_estimates)
    return test_pass_fail_rate
  def test_pass_fail_rate(self,our_estimates):
    test = 1*(our_estimates == self.test_answers) # 1* casts to int
    pass_fail_rate = test.sum()/test.size
    return pass_fail_rate
  def train(self):
    percentage_points_change_threshold = .1
    percentage_points_change = 1 # initialize
    test_pass_fail_rate_old = 0 # initialize
    pass_number = 0 # initialize
    while percentage_points_change > percentage_points_change_threshold:
      self.randomize_images() # not sure if we should
      self.train_once()
      test_pass_fail_rate = self.test()
      percentage_points_change = 100*(test_pass_fail_rate - test_pass_fail_rate_old)
      print(f'accuracy for pass {pass_number:d}: {test_pass_fail_rate:.1%}')
      test_pass_fail_rate_old = test_pass_fail_rate
      pass_number += 1
  def train_once(self):
    for batch_number in range(0,self.n_batches):
      input_array = np.take( # batch_size x n_pixels
        self.train_images,
        batch_number*self.batch_size+np.arange(0,self.batch_size),
        axis=0
      )
      fp = self.forward_pass(input_array,batch_number)
      bp = self.backward_pass(input_array,batch_number,fp)
      if np.mod(batch_number,400)==0:
        display(self.image_of_weights_list())
  def forward_pass(self,input_array,batch_number,is_test=False):
    fp = dict()
    logits = self.logitser(input_array) # batch_size x n_classes
    if is_test:
      answers = self.test_answers
    else:
      answers = self.train_answers
    fp['answers'] = np.take( # correct answers ... batch_size x 1
      answers,
      batch_number*self.batch_size+np.arange(0,self.batch_size)
    )
    logits_answers = np.empty(self.batch_size) # initialize
    for i,answer in enumerate(fp['answers']):
      logits_answers[i] = logits[i,answer] # correct logits ... batch_size x 1
      # print(fp['answers'][i])
      # display(self.image_viewer(self.scale_image_for_display(input_array[i,:].reshape((28,28))).astype(np.uint8)))
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
    bp['delta_w'] = np.zeros(self.W.shape)
    bp['delta_b'] = np.zeros(self.B.shape)
    # (1.18)
    for i in range(0,self.batch_size): # each image
      # display(self.image_viewer(self.scale_image_for_display(input_array[i,:].reshape((28,28))).astype(np.uint8)))
      # print(f'answer: {fp["answers"][i]:d}')
      for j in range(0,self.n_classes): # each class
        # print('p_all[i,j]')
        # print(fp['p_all'][i,j])
        if j == fp['answers'][i]: # j == a
          dX_dl[i,j] = -(1-fp['p_all'][i,j])
        else: # j != a
          dX_dl[i,j] = fp['p_all'][i,j]
        # print(f'perceptron: {int(j):d}, dX_dl: {dX_dl[i,j]:5.2f}')
      bp['delta_w'] = bp['delta_w']-self.learning_rate*np.outer(input_array[i,:],dX_dl[i,:])
      bp['delta_b'] = bp['delta_b']-self.learning_rate*dX_dl[i,:]
    # bp['delta_w'] = -self.learning_rate*input_array.T@dX_dl # contracts batch_size axis
    # bp['delta_b'] = -np.sum(self.learning_rate*dX_dl,axis=0) # sums batch_size axis
    # update weights and biases
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
  def image_unreshaper(self,img):
    side = int(np.sqrt(self.n_pixels))
    return img.reshape((img.shape[0],side,side))
  def image_viewer(self,image):
    return Image.fromarray(image,mode='L')
  def image_of_weights(self,digit=0):
    image_array = np.squeeze(
        self.image_unreshaper(
          np.array([self.W[:,digit]])))
    image_array = scale_image_for_display(image_array)
    return Image.fromarray(
      image_array.astype(np.uint8),
      mode='L'
    )
  def image_of_weights_list(self):
    side = np.int(np.sqrt(self.n_pixels))
    pre_concatenated_image = self.image_unreshaper(self.W.T)
    pre_concatenated_image = self.scale_image_for_display(pre_concatenated_image)
    pre_concatenated_image = np.insert(
      pre_concatenated_image,
      side,
      0,
      axis=2
    )
    concatenated_image = pre_concatenated_image.reshape((side,self.n_classes*(side+1)))
    return Image.fromarray(
      concatenated_image.astype(np.uint8),
      mode='L'
    )
  def scale_image_for_display(self,image):
    return 255*image/image.max()
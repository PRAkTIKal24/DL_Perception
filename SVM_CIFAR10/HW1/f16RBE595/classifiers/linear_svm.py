import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg, flum=False):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  Wt = np.transpose(W)
  Xt = np.transpose(X)
  #print ("Wt",Wt)
  #print("xt",Xt)
  delta = 1.0
  dW = np.zeros(W.shape)
  num_train = Xt.shape[0]
  #print ("num_train", num_train)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  
  scorest = np.dot(W,X) 
  #print("%%",np.shape(scores))
  scores = np.transpose(scorest)

  correct_class_score = scores[np.arange(num_train), y]

  margins = np.maximum(0, scores.T - correct_class_score + delta)

  loss = (np.sum(margins) - num_train) / num_train
  loss += 0.5 * reg * np.sum(W*W)

  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  slopes = np.zeros((margins.shape))
  slopes[margins>0] = 1

  slopes[y, range(num_train)] -= np.sum(margins>0, axis=0)
  
  dWt = np.dot(Xt.T, slopes.T) / float(num_train)
  dWt += reg * Wt
  dW = np.transpose(dWt)

  pass  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  
  
  return loss, dW

  

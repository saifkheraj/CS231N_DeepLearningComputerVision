import numpy as np
from random import shuffle
import copy

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #print dW.shape
  # compute the loss and the gradient
  num_classes = W.shape[1] # total classes, in this case there are 10
  num_train = X.shape[0] # n examples in this case there 49000 training examples
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # 1 x K (classes) in this assignment
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue  #not to use correct class score, only to subtract it
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]=dW[:,j]+X[i].T
        #margin violated so add it
        dW[:,y[i]]=-X[i].T + dW[:,y[i]]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW=dW/num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) # L2 loss  + data loss
  dW=dW+(reg*W)
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass

  scores=np.dot(X,W)
  correct_scores=scores[np.arange(len(scores)), y]
  correct_scores=correct_scores.reshape(correct_scores.shape[0],1)
  #print correct_scores.shape
  margins=(scores-correct_scores)+1
  margins[np.arange(len(scores)), y]=0
  margins[margins<0]=0
  
  margins_grad=copy.copy(margins) # copy margins for computing gradient later
  
  #print margins
  
  loss=np.sum(margins)/float(X.shape[0])
  
  
  loss += 0.5 * reg * np.sum(W * W)
  
  
  margins_grad[margins_grad<0]=0
  margins_grad[margins_grad>0]=1
  
  
  
  #print margins_grad
  
  
  margins_violated=np.sum(margins_grad,axis=1) #to see margins violated
  #print X.shape
  #print margins_grad.shape
  margins_grad[np.arange(len(margins_grad)), y]=-1* margins_violated
  dW=(np.dot(margins_grad.T,X)).T
  dW=dW/float(X.shape[0])
  dW+=reg*W
  #print margins_grad
  #print margins
  #print correct_scores.shape
  #print scores.shape
  #print y
  #print scores
  #print scores[0,y[0]]
  #print scores[:,y]
  
  
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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

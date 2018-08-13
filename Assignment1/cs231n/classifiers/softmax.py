import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  ntrain_examples=X.shape[0]
  nclasses=dW.shape[1]
  #print nclasses
  
  for i in xrange(ntrain_examples):

  	  scores=X[i].dot(W)
  	  scores=scores-np.max(scores) #clipping according due numeric stability
  	  scores=np.exp(scores)/np.sum(np.exp(scores))
  	  #print scores[y[i]] #correct class score
  	  loss=loss-np.log(scores[y[i]])
  	  
  	  for j in xrange(nclasses):
  	  	  #print j
  	  	  #print y[i].shape
  	  	  if(j==y[i]):
  	  	  	  dW[:,j]=dW[:,j]+ ( -X[i,:] + (scores[j] * X[i,:])	)
  	  	  	  
  	  	  else:
  	  	  	  #dW[:,j]=dW[:,j]+(-1 + scores[j]) * X[i,:]	  
  	  	  	  dW[:,j]=dW[:,j]+ ( (scores[j]) * X[i,:] ) 
  	  	  
  	  
  pass

  dW=dW/float(ntrain_examples)
  
  loss=loss/float(ntrain_examples)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W) # dataloss+regularizationloss
  dW=dW+(reg*W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  
  scores=np.dot(X,W)
  maxi=np.max(scores,axis=1)
  maxi=maxi.reshape(maxi.shape[0],1)
  scores=scores-maxi
  scores=np.exp(scores)
  sum_rows_score=np.sum(scores,axis=1)
  sum_rows_score=sum_rows_score.reshape(sum_rows_score.shape[0],1)
  scores=scores/(sum_rows_score)
  
  correct_scores=scores[np.arange(len(scores)), y]
  correct_scores=correct_scores.reshape(correct_scores.shape[0],1)
  loss=-np.sum(np.log(correct_scores))/float(X.shape[0])
  loss += 0.5 * reg * np.sum(W * W)
  
  
  scores[np.arange(len(scores)), y]=scores[np.arange(len(scores)), y]-1
  #print scores.T.shape
  #print X.shape
  #dW=
  dW=np.dot((scores.T),X).T
  dW=dW/X.shape[0]
  dW = dW+(reg * W)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


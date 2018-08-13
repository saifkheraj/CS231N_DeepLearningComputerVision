import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  out1=np.dot(prev_h,Wh)
  out2=np.dot(x,Wx)
  
  out3=out1+out2+b
  h=np.tanh(out3)
  
  
  
  next_h=h
  cache=(Wh,prev_h,out3,Wx,x)


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  
  Wh,prev_h,out3,Wx,x=cache
  dout3=(1 - np.tanh(out3)**2)  * dnext_h
  #print dout3.shape #NXH
  dout1=dout3 
  dout2=dout3
  
  
  #print dout1.shape
  #print Wh.shape
  #print prev_h.shape
  
  
  db=np.sum(dout3,axis=0)
  dprev_h=np.dot(dout1,Wh.T)
  
  dWh=np.dot(prev_h.T,dout1)
  
  #dWh=np.dot(prev_h,dout1.T)
  #print dWh.shape
  #print dprev_h.shape
  #print Wx.shape

  #print dout2.shape
  
  
  dx=np.dot(dout2,Wx.T)
  
  #print x.shape
  #print dout2.shape
  #print Wx.shape
  dWx=np.dot(x.T,dout2)
  #print dWx.shape
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  X=x.transpose(1,0,2) #now T x N X D
  h_store=[]
  cache={}
  for t in np.arange(X.shape[0]):
  	  #print h0
  	  #print X[t].shape
  	  
  	  str_c="cache"+str(t)
  	  h0,c=rnn_step_forward(X[t], h0, Wx, Wh, b)
  	  cache[str_c]=c
  	  h_store.append(h0)
  	  
  	  #print h0
  	  #X=
		#rnn_step_forward(x, prev_h, Wx, Wh, b)
  #print "shape: ",np.array(h_store).shape	  
  #print "check",np.array(h_store).transpose(1,0,2).shape
  h=np.array(h_store).transpose(1,0,2)
  
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  
  #rnn_step_backward(dnext_h, cache)
  #print "dh",dh.shape
  dH=dh.transpose(1,0,2)
  #print dH.shape
  T,N,H=dH.shape
  #Wh,prev_h,out3,Wx,x=cache['cache0']
  #print x.shape
  #T,N,H=dH.shape
  #db=np.zeros((H))
  
  #dWh=np.zeros((H,H))
  #dWx=np.zeros((H,x.shape[1]))
  #dh0=np.zeros((N,H))
  
  str_c="cache"+str(T-1)
  Wh,prev_h,out3,Wx,x=cache[str_c]
  
  dx=np.zeros((T,N,x.shape[1]))
  
  dX, dh0, dWx, dWh, db=rnn_step_backward(dH[T-1], cache[str_c])
  dx[T-1]=dX
  #print dWx.shape
  for t in np.arange(T-2,-1,-1):
  	  dh0=dh0+dH[t]
  	  str_c="cache"+str(t)
  	  dX, dh0, dWX, dWH, dB=rnn_step_backward(dh0, cache[str_c])
  	  dx[t]=dX
  	  db+=dB
  	  dWx+=dWX
  	  dWh+=dWH
  	  
  #print cache.keys()

  dx=dx.transpose(1,0,2)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  
  #print x.shape
  #similar to word2vec just have to pick particular row of Weights corresponding
  #to index , x contains indexes and each could be though of as one hot encoding
  #so we pick particular row of W
  
  #print W[x,:]
  out=W[x,:]
  cache=(x,W)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x,W=cache
  dW=np.zeros(W.shape)
  np.add.at(dW,x,dout) #just look up
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  #print Wx.shape
  #print Wh.shape
  #print b.shape
  #print prev_h.shape
  #print x.shape
  #print prev_c.shape
  
  #step1
  iout=np.dot(x,Wx) #N x 4H
  hout=np.dot(prev_h,Wh) # N x 4H
  out1=iout+hout+b # N x 4H
  
  #step2
  
  #print "out",out1.shape
  H=prev_h.shape[1]
  #print out1
  sepi=out1[:,:H]  # N x H
  sepf=out1[:,H:2*H]  #NxH
  sepo=out1[:,2*H:3*H] #NxH
  sepg=out1[:,3*H:4*H] #NxH
  
  i=sigmoid(sepi) #i
  f=sigmoid(sepf) #f 
  o=sigmoid(sepo) # o
  g=np.tanh(sepg) #g
  
  
  
  #step3
  
  o1=i*g    #N x H
  o2=f*prev_c #N x H
  next_c=o1+o2 #N x H
  #print next_c.shape
  #print next_c
  
  
  tan_c=np.tanh(next_c)
  next_h=o*tan_c
  cache=(o,next_c,f,tan_c,g,prev_c,i,sepi,sepf,sepo,sepg,x,Wx,Wh,prev_h)
  
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  
  #starting back from step 3 done above
  o,next_c,f,tan_c,g,prev_c,i,sepi,sepf,sepo,sepg,x,Wx,Wh,prev_h=cache
  
  dtan_c=o*dnext_h
  dnext_C=(1 - np.tanh(next_c)**2) * dtan_c
  
  do1=dnext_C + dnext_c  #add operation so just passing
  do2=dnext_C+dnext_c #coming from 2 ways
  
  dprev_c=(do2*f) 
  
  
  do=tan_c * dnext_h
  di=(g*do1)
  df=prev_c*do2
  dg=i*do1
  
  #step2
  dsepi=di*(sigmoid(sepi)*(1-sigmoid(sepi)))
  dsepf=df*(sigmoid(sepf)*(1-sigmoid(sepf)))
  dsepo=do*(sigmoid(sepo)*(1-sigmoid(sepo)))
  dsepg=dg*(1 - np.tanh(sepg)**2)
  
  #each is N x H so we need to stack them up during back prob
  
  dout1=np.hstack((dsepi,dsepf,dsepo,dsepg)) # should be N x 4H
  #print dout1.shape
  
  
  #Back to step 1 (doing according to flowgraph I made)
  db=np.sum(dout1,axis=0)
  #dWh=np.dot(dout1,)
  #print x.shape
  #print Wx.shape
  
  dx=np.dot(dout1,Wx.T)
  
  dWx=np.dot(x.T,dout1)
  dWh=np.dot(prev_h.T,dout1)
  dprev_h=np.dot(dout1,Wh.T)
  
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  
  X=x.transpose(1,0,2) #now T x N X D
  prev_c=np.zeros(h0.shape)
  h_store=[]
  cache={}
  for t in np.arange(X.shape[0]):
  	  
  	  str_c="cache"+str(t)
  	  h0, next_c, c=lstm_step_forward(X[t], h0,prev_c ,Wx, Wh, b)
  	  prev_c=next_c
  	  cache[str_c]=c
  	  h_store.append(h0)

  h=np.array(h_store).transpose(1,0,2)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  
  dH=dh.transpose(1,0,2)
  #print dH.shape
  T,N,H=dH.shape  
  str_c="cache"+str(T-1)
  
  #just to get x and its dimension so that dx could be initialized ,
  #rest variables are not needed only x is needed
  o,next_c,f,tan_c,g,prev_c,i,sepi,sepf,sepo,sepg,x,Wx,Wh,prev_h=cache[str_c]
  
  dx=np.zeros((T,N,x.shape[1]))
  dc=np.zeros((N,H))
  
  dX, dh0, dC, dWx, dWh, db=lstm_step_backward(dH[T-1], dc,cache[str_c])
  
  dx[T-1]=dX
  #print dWx.shape
  for t in np.arange(T-2,-1,-1):
  	  dh0=dh0+dH[t]
  	  str_c="cache"+str(t)
  	  dX, dh0, dC,dWX, dWH, dB=lstm_step_backward(dh0,dC, cache[str_c])
  	  dx[t]=dX
  	  db+=dB
  	  dWx+=dWX
  	  dWh+=dWH
  	  
  
  dx=dx.transpose(1,0,2)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx


from __future__ import division
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
  for i in xrange(X.shape[0]):
    score = X[i,:].dot(W)
    score -= score.max()
    # loss += -np.log(np.exp(score[y[i]])/np.sum(np.exp(score)))
    loss += - np.log(P(score, y[i]))
    # Loop over classes
    dscore = P(score, range(W.shape[1]))
    for j in xrange(W.shape[1]):
      # Correct class
    #   if j == y[i]:
    #     dscore[j] -= 1
    # # print X[i,:].shape, dscore.shape
    #   dW[:,j] += X[i,:] * dscore[j]
      dW[:,j] += X[i,:] * (P(score, j) - (j == y[i]))
    # dW += np.outer(X[i,:], dscore)
        # pass
        
  loss /= X.shape[0]
  loss += 0.5*reg*np.sum(W*W)
  dW /= X.shape[0]
  dW += reg * W
  # pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
def P(arr, i):
  return (np.exp(arr[i])/np.sum(np.exp(arr)))

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
    Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # score (N,C)
  score = X.dot(W)
  exp_score = np.exp(score)
  prob = exp_score / exp_score.sum(axis=1)[:,None]
  pk = prob[xrange(num_train), y]
  loss = -np.log(pk).sum()
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dLdf = prob
  dLdf[xrange(num_train), y] -= 1
  dW = X.T.dot(dLdf)
  dW /= num_train
  dW += reg * W
  # pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


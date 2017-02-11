import numpy as np
from random import shuffle


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        temp = 0
        for j in xrange(num_classes):
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if j == y[i]:
                continue
            if margin > 0:
                dW[:, j] += X[i]
                loss += margin
                temp += 1
        dW[:, y[i]] -= temp * X[i]

    dW /= num_train
    # dW += reg * np.linalg.norm(W)
    dW += reg * W
    # dW += reg * np.sum(W)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
      Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    loss = 0.0
    dW = np.zeros(W.shape, dtype=float)  # initialize the gradient as zero

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    ##########################################################################

    score = X.dot(W)
    correct_score = np.diag(score.T[y])
    margin = score - correct_score[:, None] + 1
    idx = margin < 0
    margin[idx] = 0
    margin[xrange(X.shape[0]), y] = 0
    temp_sum = (margin).sum(axis=1)
    loss = temp_sum.mean() + 0.5 * reg * np.sum(W * W)
    # pass
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    ##########################################################################

    # temp_grad = -X * (idx.sum(axis=1)-1)[:,None]
    # temp_grad = X.mean(axis=0)
    # dW[:,:] = temp_grad[:,None]
    # dW[:,:] = X.mean(axis=0)[:,None]
    # dW = dW.T
    # dW[np.arange(X.shape[0]), y] *= -idx.sum(axis=1) + 1

    ########
    # Imagine if there's only 1 sample
    # Xtest = X[0,:]
    # ytest = y[0]
    # dW[:,[x for x in xrange(W.shape[1]) if x != ytest]] =
    # dW[range(W.shape[0]), ytest] = -(idx[0].sum(axis=1) - idx[0,ytest]) * Xtest

    idx = margin > 0
    idx = idx.astype(float)
    wrong_class = idx.sum(axis=1)
    idx[xrange(X.shape[0]), y] = -wrong_class
    dW = X.T.dot(idx)
    dW /= float(X.shape[0])
    dW += reg * W

    # pass
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW

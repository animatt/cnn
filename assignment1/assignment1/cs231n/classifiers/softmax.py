import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_examples = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]

  for row, ind in zip(X, y):
      score = row.dot(W)
      exp_scores = np.exp(score)
      loss += -score[ind] + np.log(np.sum(exp_scores))

  loss /= num_examples
  loss += reg * np.sum(W ** 2)

  for i in range(num_features):
      for j in range(num_classes):
          dW[i, j] += -np.sum(X[y == j, i])
          dW[i, j] += np.sum(1 / np.sum(
              np.exp(X.dot(W)), keepdims=True, axis=1
          ) * np.exp(X.dot(W[:, j:j + 1])) * X[:, i:i + 1])
          dW[i, j] /= num_examples
          dW[i, j] += 2 * reg * W[i, j]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  N, D = X.shape
  num_classes = W.shape[1]
  exp_scores = np.exp(X.dot(W))
  csummed_exp_scores = np.sum(exp_scores, keepdims=True, axis=1)
  inside = np.log(exp_scores[range(N), y].reshape((-1, 1)) / csummed_exp_scores)
  loss = -1 / N * np.sum(inside) + reg * np.sum(W ** 2)
  
  dW = - 1 / N * X.T.dot(y.reshape((-1, 1)) == range(num_classes))
  dW += 1 / N * X.T.dot(exp_scores / csummed_exp_scores) + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


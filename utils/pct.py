import numpy as np
from matplotlib import pyplot as plt

class PCT:
  def __init__(self, no_of_feature,  W_inital: np.ndarray = None) -> None:
    if W_inital is None:
      self.__W = np.random.random((no_of_feature + 1, 1))

    else:
      self.__W = W_inital

  def fit(
      self, 
      X: np.ndarray, 
      Y: np.ndarray, 
      learning_rate, 
      max_iteration, 
      convergence_threshold = 0.001,
    ):

    X_bar = np.hstack((np.ones((X.shape[0], 1)), X))

    dataset_length = X_bar.shape[0]
    for i in range(max_iteration):
      newW = self.__W
      for j in range(dataset_length):
        x = X_bar[j]
        y = Y[j]
        sign = np.sign(x.dot(self.__W))
        if sign != y:
          newW = newW + learning_rate * np.expand_dims(x, 1) * y

      if np.sum(np.abs(newW - self.__W)) / np.sum(np.abs(self.__W)) < convergence_threshold:
        self.__W = newW
        break

      self.__W = newW
    return i

  def predict(self, X: np.ndarray) -> np.ndarray:
    X_bar = np.hstack((np.ones((X.shape[0], 1)), X))
    return np.sign(X_bar.dot(self.__W))

  def getW(self) -> np.ndarray:
    return self.__W.copy()
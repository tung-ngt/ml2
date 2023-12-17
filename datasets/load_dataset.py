import numpy as np
import pandas as pd
import os

def iris(shuffle=False) -> tuple[pd.DataFrame, pd.DataFrame]:
  current = os.path.dirname(os.path.abspath(__file__))
  data = pd.read_csv(current + "/iris.data", header=None)
  feature_column = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  if (shuffle):
    data = data.sample(frac=1).reset_index(drop=True)
  data.columns = [*feature_column, "class"]
  return data, data[feature_column]


def rice(shuffle=False) -> tuple[pd.DataFrame, pd.DataFrame]:
  current = os.path.dirname(os.path.abspath(__file__))
  data = pd.read_csv(current + "/Rice_Cammeo_Osmancik.csv", header=None)
  if (shuffle):
    data = data.sample(frac=1).reset_index(drop=True)
  feature_column = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove_length']
  data.columns = [*feature_column, "class"]
  
  return data, data[feature_column]

def wine(shuffle=False):
  current = os.path.dirname(os.path.abspath(__file__))
  data = pd.read_csv(current + "/wine.data", header=None)
  if (shuffle):
    data = data.sample(frac=1).reset_index(drop=True)
  feature_column = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
  data.columns = ["class", *feature_column]
  return data, data[feature_column]
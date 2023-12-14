import pandas as pd
import math

def split_data(data: pd.DataFrame, ratio: tuple[pd.DataFrame]) -> list[pd.DataFrame]:
  data_list = []
  start = 0
  data_rows = len(data)
  for r in ratio[:-1]:
    end = start + math.floor(data_rows * r)
    data_list.append(data[start: end])
    start = end

  data_list.append(data[start: data_rows])

  return data_list
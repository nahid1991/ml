import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd

df = pd.read_csv('train.csv')
print(df.head(15))
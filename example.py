import pandas as panpan
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn import linear_model

iris_csv_path = "iris.csv"
iris_df = panpan.read_csv(iris_csv_path)

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from mlxtend.plotting import plot_linear_regression
from sklearn.preprocessing import PolynomialFeatures
import json

data_movies = pd.read_csv('tmdb_5000_movies_train.csv')
data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')
df = pd.DataFrame(data_ab)

df.drop('homepage', axis=1, inplace=True)
df.drop('original_title', axis=1, inplace=True)
df.drop('overview', axis=1, inplace=True)
df.drop('tagline', axis=1, inplace=True)
df.drop('title_x', axis=1, inplace=True)
df.drop('title_y', axis=1, inplace=True)
df.drop('movie_id', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
df.drop('status', axis=1, inplace=True)
df.dropna(how='any', inplace=True)

Y = df.iloc[:, -4]
X = df.iloc[:, df.keys() != 'vote_average']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
df_xtrain = pd.DataFrame(X_train)
y_train = y_train.tolist()
df_xtest = pd.DataFrame(X_test)
y_test = y_test.tolist()


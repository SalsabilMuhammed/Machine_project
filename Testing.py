import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures
import json

data_movies = pd.read_csv('tmdb_5000_movies_train.csv')
# data_movies.dropna(how='any', inplace=True)
data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')
df = pd.DataFrame(data_ab)

# drop columns
df.drop('homepage', axis=1, inplace=True)
df.drop('release_date', axis=1, inplace=True)
df.drop('original_title', axis=1, inplace=True)
df.drop('overview', axis=1, inplace=True)
df.drop('tagline', axis=1, inplace=True)
df.drop('title_x', axis=1, inplace=True)
df.drop('title_y', axis=1, inplace=True)
df.drop('movie_id', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
df.drop('status', axis=1, inplace=True)
df.dropna(how='any', inplace=True)
# normalize budget and revenue
df = df[df.budget != 0]
budget_max = np.max(df['budget'])
budget_min = np.min(df['budget'])
dif = budget_max - budget_min
df['budget'] = df['budget'] - budget_min
df['budget'] = (df['budget'] / dif) * 3

revenue_min = np.min(df['revenue'])
revenue_max = np.max(df['revenue'])
df['revenue'] = df['revenue'] - revenue_min
df['revenue'] = (df['revenue'] / (revenue_max - revenue_min)) * 3

# normalize popularity
popularity_min = np.min(df['popularity'])
popularity_max = np.max(df['popularity'])
df['popularity'] = df['popularity'] - popularity_min
df['popularity'] = (df['popularity'] / (popularity_max - popularity_min)) * 3

# normalize runtime
runtime_min = np.min(df['runtime'])
runtime_max = np.max(df['runtime'])
df['runtime'] = df['runtime'] - runtime_min
df['runtime'] = (df['runtime'] / (runtime_max - runtime_min)) * 3

# normalize vote count
vote_count_min = np.min(df['vote_count'])
vote_count_max = np.max(df['vote_count'])
df['vote_count'] = df['vote_count'] - vote_count_min
df['vote_count'] = (df['vote_count'] / (vote_count_max - vote_count_min)) * 3

print(df.keys())

X = df.iloc[:, df.keys() != 'vote_average']
Y = df.iloc[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
X_df = pd.DataFrame(X_train)
Y_df = pd.DataFrame(y_train)
vote_average = Y_df['vote_average'].tolist()
print(vote_average)
# Pre_processing Train Data
# genres pre_processing
genres_count = {}
genres_score = {}
genres = X_df['genres'].tolist()
for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] = 0
        genres_score[j['name']] = 0

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] += 1

for i in range(len(genres)):
    for j in json.loads(genres[i]):
        print(vote_average[i])
        genres_score[j['name']] += vote_average[i]

for i in genres_score.keys():
    genres_score[i] = genres_score[i] / genres_count[i]

genres_encoded = []
for i in genres:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += genres_score[j['name']]
        cnt += 1
    if cnt != 0:
        genres_encoded.append(tmp / cnt)
    else:
        genres_encoded.append(tmp)
X_df['genres'] = genres_encoded

# Normalize Genres
genres_min = np.min(X_df['genres'])
genres_max = np.max(X_df['genres'])
X_df['genres'] = X_df['genres'] - genres_min
X_df['genres'] = ((X_df['genres'] / (genres_max - genres_min)) * 3)

# keywords pre_processing
keywords_count = {}
keywords_score = {}
keywords = X_df['keywords'].tolist()
for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] = 0
        keywords_score[j['name']] = 0

for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] += 1

for i in range(len(keywords)):
    for j in json.loads(keywords[i]):
        keywords_score[j['name']] += vote_average[i]

for i in keywords_score.keys():
    keywords_score[i] = keywords_score[i] / keywords_count[i]

keywords_encoded = []
for i in keywords:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += keywords_score[j['name']]
        cnt += 1
    if cnt != 0:
        keywords_encoded.append(tmp / cnt)
    else:
        keywords_encoded.append(tmp)
X_df['keywords'] = keywords_encoded

# Normalize keyword
keywords_min = np.min(X_df['keywords'])
keywords_max = np.max(X_df['keywords'])
X_df['keywords'] = X_df['keywords'] - keywords_min
X_df['keywords'] = ((X_df['keywords'] / (keywords_max - keywords_min)) * 3)

# production companies pre_processing
prodcomp_count = {}
prodcomp_score = {}
prodcomp = X_df['production_companies'].tolist()
for i in prodcomp:
    for j in json.loads(i):
        prodcomp_count[j['name']] = 0
        prodcomp_score[j['name']] = 0

for i in prodcomp:
    for j in json.loads(i):
        prodcomp_count[j['name']] += 1

for i in range(len(prodcomp)):
    for j in json.loads(prodcomp[i]):
        prodcomp_score[j['name']] += vote_average[i]

for i in prodcomp_score.keys():
    prodcomp_score[i] = prodcomp_score[i] / prodcomp_count[i]

prodcomp_encoded = []
for i in prodcomp:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcomp_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcomp_encoded.append(tmp / cnt)
    else:
        prodcomp_encoded.append(tmp)
X_df['production_companies'] = prodcomp_encoded

# Normalize Production Companies
prodcomp_min = np.min(X_df['production_companies'])
prodcomp_max = np.max(X_df['production_companies'])
X_df['production_companies'] = X_df['production_companies'] - prodcomp_min
X_df['production_companies'] = ((X_df['production_companies'] / (prodcomp_max - prodcomp_min)) * 3)

# production countries pre_processing
prodcountry_count = {}
prodcountry_score = {}
prodcountry = X_df['production_countries'].tolist()
for i in prodcountry:
    for j in json.loads(i):
        prodcountry_count[j['name']] = 0
        prodcountry_score[j['name']] = 0

for i in prodcountry:
    for j in json.loads(i):
        prodcountry_count[j['name']] += 1

for i in range(len(prodcountry)):
    for j in json.loads(prodcountry[i]):
        prodcountry_score[j['name']] += vote_average[i]

for i in prodcountry_score.keys():
    prodcountry_score[i] = prodcountry_score[i] / prodcountry_count[i]

prodcountry_encoded = []
for i in prodcountry:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcountry_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcountry_encoded.append(tmp / cnt)
    else:
        prodcountry_encoded.append(tmp)
X_df['production_countries'] = prodcountry_encoded

# Normalize Production Country
prodcountry_min = np.min(X_df['production_countries'])
prodcountry_max = np.max(X_df['production_countries'])
X_df['production_countries'] = X_df['production_countries'] - prodcountry_min
X_df['production_countries'] = ((X_df['production_countries'] / (prodcountry_max - prodcountry_min)) * 3)

# spoken languages pre_processing
spokenlang_count = {}
spokenlang_score = {}
spokenlang = X_df['spoken_languages'].tolist()
for i in spokenlang:
    for j in json.loads(i):
        spokenlang_count[j['name']] = 0
        spokenlang_score[j['name']] = 0

for i in spokenlang:
    for j in json.loads(i):
        spokenlang_count[j['name']] += 1

for i in range(len(spokenlang)):
    for j in json.loads(spokenlang[i]):
        spokenlang_score[j['name']] += vote_average[i]

for i in spokenlang_score.keys():
    spokenlang_score[i] = spokenlang_score[i] / spokenlang_count[i]

spokenlang_encoded = []
for i in spokenlang:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += spokenlang_score[j['name']]
        cnt += 1
    if cnt != 0:
        spokenlang_encoded.append(tmp / cnt)
    else:
        spokenlang_encoded.append(tmp)
X_df['spoken_languages'] = spokenlang_encoded

# Normalize Spokenlang
spokenlang_min = np.min(X_df['spoken_languages'])
spokenlang_max = np.max(X_df['spoken_languages'])
X_df['spoken_languages'] = X_df['spoken_languages'] - spokenlang_min
X_df['spoken_languages'] = ((X_df['spoken_languages'] / (spokenlang_max - spokenlang_min)) * 3)

# cast pre_processing
cast_count = {}
cast_score = {}
cast = X_df['cast'].tolist()
for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] = 0
        cast_score[j['name']] = 0

for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] += 1

for i in range(len(cast)):
    for j in json.loads(cast[i]):
        cast_score[j['name']] += vote_average[i]

for i in cast_score.keys():
    cast_score[i] = cast_score[i] / cast_count[i]

cast_encoded = []
for i in cast:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += cast_score[j['name']]
        cnt += 1
    if cnt != 0:
        cast_encoded.append(tmp / cnt)
    else:
        cast_encoded.append(tmp)
X_df['cast'] = cast_encoded

# Normalize Cast
cast_min = np.min(X_df['cast'])
cast_max = np.max(X_df['cast'])
X_df['cast'] = X_df['cast'] - cast_min
X_df['cast'] = ((X_df['cast'] / (cast_max - cast_min)) * 3)

# crew pre_processing
crew_dep_count = {}
crew_dep_score = {}
crew_job_count = {}
crew_job_score = {}
crew_name_count = {}
crew_name_score = {}
crew = X_df['crew'].tolist()
for i in crew:
    for j in json.loads(i):
        crew_name_count[j['name']] = 0
        crew_name_score[j['name']] = 0

for i in crew:
    for j in json.loads(i):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_name_count[j['name']] += 1

for i in range(len(crew)):
    for j in json.loads(crew[i]):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_name_score[j['name']] += vote_average[i]

for i in crew:
    for j in json.loads(i):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_name_score[j['name']] = crew_name_score[j['name']] / crew_name_count[j['name']]

# for i in crew_name_score.keys():
#     crew_name_score[i] = crew_name_score[i]/crew_name_count[i]

crew_encoded = []
for i in crew:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += crew_name_score[j['name']]
        cnt += 1
    if cnt != 0:
        crew_encoded.append(tmp / cnt)
    else:
        crew_encoded.append(tmp)
X_df['crew'] = crew_encoded

# Normalize Crew
crew_min = np.min(X_df['crew'])
crew_max = np.max(X_df['crew'])
X_df['crew'] = X_df['crew'] - crew_min
X_df['crew'] = ((X_df['crew'] / (crew_max - crew_min)) * 3)

# language pre_processing
lang = df['original_language'].tolist()
lang_count = {}
lang_score = {}
for i in lang:
    lang_count[i] = 0
    lang_score[i] = 0
for i in lang:
    lang_count[i] += 1
cnt = 0
for i in lang:
    lang_score[i] += vote_average[cnt]
    cnt += 1
for i in lang_count.keys():
    lang_score[i] = lang_score[i] / lang_count[i]
lang_encoded = []
for i in lang:
    lang_encoded.append(lang_score[i])
X_df['original_language'] = lang_encoded
# Normalize Language
lang_min = np.min(X_df['original_language'])
lang_max = np.max(X_df['original_language'])
X_df['original_language'] = X_df['original_language'] - lang_min
X_df['original_language'] = (X_df['original_language'] / (lang_max - lang_min)) * 3

print(X_df.corr())

# Preproccing Testing Data
X_df_Test = pd.DataFrame(X_test)

# Preprocessing genres
genres_encoded = []
for i in genres:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += genres_score[j['name']]
        cnt += 1
    if cnt != 0:
        genres_encoded.append(tmp / cnt)
    else:
        genres_encoded.append(tmp)
X_df_Test['genres'] = genres_encoded

# Normalize Genres
X_df_Test['genres'] = X_df_Test['genres'] - genres_min
X_df_Test['genres'] = ((X_df_Test['genres'] / (genres_max - genres_min)) * 3)

# Preprocessing Production Country
prodcountry_encoded = []
for i in prodcountry:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcountry_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcountry_encoded.append(tmp / cnt)
    else:
        prodcountry_encoded.append(tmp)
X_df_Test['production_countries'] = prodcountry_encoded

# Normalize Production Country
X_df_Test['production_countries'] = X_df_Test['production_countries'] - prodcountry_min
X_df_Test['production_countries'] = ((X_df_Test['production_countries'] / (prodcountry_max - prodcountry_min)) * 3)

# Preprossing Keyword
keywords_encoded = []
for i in keywords:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += keywords_score[j['name']]
        cnt += 1
    if cnt != 0:
        keywords_encoded.append(tmp / cnt)
    else:
        keywords_encoded.append(tmp)
X_df_Test['keywords'] = keywords_encoded

# Normalize KeyWord

X_df_Test['keywords'] = X_df_Test['keywords'] - keywords_min
X_df_Test['keywords'] = ((X_df_Test['keywords'] / (keywords_max - keywords_min)) * 3)

# Preprossing Production Companies
prodcomp_encoded = []
for i in prodcomp:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcomp_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcomp_encoded.append(tmp / cnt)
    else:
        prodcomp_encoded.append(tmp)
X_df_Test['production_companies'] = prodcomp_encoded

# Normalize Production Companies
X_df_Test['production_companies'] = X_df_Test['production_companies'] - prodcomp_min
X_df_Test['production_companies'] = ((X_df_Test['production_companies'] / (prodcomp_max - prodcomp_min)) * 3)

# Preprossing Spoken Language
spokenlang_encoded = []
for i in spokenlang:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += spokenlang_score[j['name']]
        cnt += 1
    if cnt != 0:
        spokenlang_encoded.append(tmp / cnt)
    else:
        spokenlang_encoded.append(tmp)
X_df_Test['spoken_languages'] = spokenlang_encoded

# Normalize Spokenlang
X_df_Test['spoken_languages'] = X_df_Test['spoken_languages'] - spokenlang_min
X_df_Test['spoken_languages'] = ((X_df_Test['spoken_languages'] / (spokenlang_max - spokenlang_min)) * 3)

# Preprossing Cast
cast_encoded = []
for i in cast:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += cast_score[j['name']]
        cnt += 1
    if cnt != 0:
        cast_encoded.append(tmp / cnt)
    else:
        cast_encoded.append(tmp)
X_df_Test['cast'] = cast_encoded

# Normalize Cast
X_df_Test['cast'] = X_df_Test['cast'] - cast_min
X_df_Test['cast'] = ((X_df_Test['cast'] / (cast_max - cast_min)) * 3)

# Preprossing Crew
crew_encoded = []
for i in crew:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += crew_name_score[j['name']]
        cnt += 1
    if cnt != 0:
        crew_encoded.append(tmp / cnt)
    else:
        crew_encoded.append(tmp)
X_df_Test['crew'] = crew_encoded

# Normalize Crew
X_df_Test['crew'] = X_df_Test['crew'] - crew_min
X_df_Test['crew'] = ((X_df_Test['crew'] / (crew_max - crew_min)) * 3)

# Preprossing Language
lang_encoded = []
for i in lang:
    lang_encoded.append(lang_score[i])
X_df_Test['original_language'] = lang_encoded
# Normalize Language
X_df_Test['original_language'] = X_df_Test['original_language'] - lang_min
X_df_Test['original_language'] = (X_df_Test['original_language'] / (lang_max - lang_min)) * 3
# drop null rows
X_df.dropna(how='any', inplace=True)
Y_df.dropna(how='any', inplace=True)
X_df_Test.dropna(how='any', inplace=True)

# Models
Xtrain = X_df.iloc[:, :]
print(X.keys())
Ytrain = Y_df.iloc[:, :]

Xtest = X_df_Test.iloc[:, :]

clf = linear_model.LinearRegression()
clf.fit(Xtrain, Ytrain)
y_hat = clf.predict(Xtest)
print("Linear " + str(metrics.r2_score(y_test, y_hat)))

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(Xtrain)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Ytrain)
prediction = poly_model.predict(poly_features.fit_transform(Xtest))
print("polynomial " + str(metrics.r2_score(y_test, prediction)))

model = linear_model.Ridge()
model.fit(Xtrain, Ytrain)
y_hat = model.predict(Xtest)
print("ridge " + str(metrics.r2_score(y_test, y_hat)))

model = linear_model.BayesianRidge()
model.fit(Xtrain, Ytrain)
y_hat = model.predict(Xtest)
print("bayes " + str(metrics.r2_score(y_test, y_hat)))

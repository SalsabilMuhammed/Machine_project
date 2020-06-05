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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
df_xtrain = pd.DataFrame(X_train)
y_train = y_train.tolist()
df_xtest = pd.DataFrame(X_test)
y_test = y_test.tolist()



#genres pre_processing
genres_count = {}
genres_score = {}
genres = df_xtrain['genres'].tolist()

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] = 0
        genres_score[j['name']] = 0

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] += 1

for i in range(len(genres)):
    for j in json.loads(genres[i]):
        genres_score[j['name']] += y_train[i]


for i in genres_score.keys():
    genres_score[i] = genres_score[i]/genres_count[i]

genres_encoded = []
for i in genres:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += genres_score[j['name']]
        cnt += 1
    if cnt != 0:
        genres_encoded.append(tmp/cnt)
    else:
        genres_encoded.append(tmp)
df_xtrain['genres'] = genres_encoded
genres_min = np.min(df_xtrain['genres'])
genres_max = np.max(df_xtrain['genres'])
df_xtrain['genres'] = df_xtrain['genres']-genres_min
df_xtrain['genres'] = ((df_xtrain['genres']/(genres_max-genres_min)))
#print(df_xtrain['genres'])


#keywords pre_processing
keywords_count = {}
keywords_score = {}
keywords = df_xtrain['keywords'].tolist()
for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] = 0
        keywords_score[j['name']] = 0

for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] += 1

for i in range(len(keywords)):
    for j in json.loads(keywords[i]):
        keywords_score[j['name']] += y_train[i]

for i in keywords_score.keys():
    keywords_score[i] = keywords_score[i]/keywords_count[i]

keywords_encoded = []
for i in keywords:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += keywords_score[j['name']]
        cnt += 1
    if cnt != 0:
        keywords_encoded.append(tmp/cnt)
    else:
        keywords_encoded.append(tmp)
df_xtrain['keywords'] = keywords_encoded

keywords_min = np.min(df_xtrain['keywords'])
keywords_max = np.max(df_xtrain['keywords'])
df_xtrain['keywords'] = df_xtrain['keywords']-keywords_min
df_xtrain['keywords'] = ((df_xtrain['keywords']/(keywords_max-keywords_min)))



#production companies pre_processing
prodcomp_count = {}
prodcomp_score = {}
prodcomp = df_xtrain['production_companies'].tolist()
for i in prodcomp:
    for j in json.loads(i):
        prodcomp_count[j['name']] = 0
        prodcomp_score[j['name']] = 0

for i in prodcomp:
    for j in json.loads(i):
        prodcomp_count[j['name']] += 1

for i in range(len(prodcomp)):
    for j in json.loads(prodcomp[i]):
        prodcomp_score[j['name']] += y_train[i]

for i in prodcomp_score.keys():
    prodcomp_score[i] = prodcomp_score[i]/prodcomp_count[i]

prodcomp_encoded = []
for i in prodcomp:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcomp_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcomp_encoded.append(tmp/cnt)
    else:
        prodcomp_encoded.append(tmp)
df_xtrain['production_companies'] = prodcomp_encoded
prodcomp_min = np.min(df_xtrain['production_companies'])
prodcomp_max = np.max(df_xtrain['production_companies'])
df_xtrain['production_companies'] = df_xtrain['production_companies']-prodcomp_min
df_xtrain['production_companies'] = ((df_xtrain['production_companies']/(prodcomp_max-prodcomp_min)))


#production countries pre_processing
prodcountry_count = {}
prodcountry_score = {}
prodcountry = df_xtrain['production_countries'].tolist()
for i in prodcountry:
    for j in json.loads(i):
        prodcountry_count[j['name']] = 0
        prodcountry_score[j['name']] = 0

for i in prodcountry:
    for j in json.loads(i):
        prodcountry_count[j['name']] += 1

for i in range(len(prodcountry)):
    for j in json.loads(prodcountry[i]):
        prodcountry_score[j['name']] += y_train[i]

for i in prodcountry_score.keys():
    prodcountry_score[i] = prodcountry_score[i]/prodcountry_count[i]

prodcountry_encoded = []
for i in prodcountry:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += prodcountry_score[j['name']]
        cnt += 1
    if cnt != 0:
        prodcountry_encoded.append(tmp/cnt)
    else:
        prodcountry_encoded.append(tmp)
df_xtrain['production_countries'] = prodcountry_encoded
prodcountry_min = np.min(df_xtrain['production_countries'])
prodcountry_max = np.max(df_xtrain['production_countries'])
df_xtrain['production_countries'] = df_xtrain['production_countries']-prodcountry_min
df_xtrain['production_countries'] = ((df_xtrain['production_countries']/(prodcountry_max-prodcountry_min)))



#spoken languages pre_processing
spokenlang_count = {}
spokenlang_score = {}
spokenlang = df_xtrain['spoken_languages'].tolist()
for i in spokenlang:
    for j in json.loads(i):
        spokenlang_count[j['name']] = 0
        spokenlang_score[j['name']] = 0

for i in spokenlang:
    for j in json.loads(i):
        spokenlang_count[j['name']] += 1

for i in range(len(spokenlang)):
    for j in json.loads(spokenlang[i]):
        spokenlang_score[j['name']] += y_train[i]

for i in spokenlang_score.keys():
    spokenlang_score[i] = spokenlang_score[i]/spokenlang_count[i]

spokenlang_encoded = []
for i in spokenlang:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += spokenlang_score[j['name']]
        cnt += 1
    if cnt != 0:
        spokenlang_encoded.append(tmp/cnt)
    else:
        spokenlang_encoded.append(tmp)
df_xtrain['spoken_languages'] = spokenlang_encoded
spokenlang_min = np.min(df_xtrain['spoken_languages'])
spokenlang_max = np.max(df_xtrain['spoken_languages'])
df_xtrain['spoken_languages'] = df_xtrain['spoken_languages']-spokenlang_min
df_xtrain['spoken_languages'] = ((df_xtrain['spoken_languages']/(spokenlang_max-spokenlang_min)))


#cast pre_processing
cast_count = {}
cast_score = {}
cast = df_xtrain['cast'].tolist()
for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] = 0
        cast_score[j['name']] = 0

for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] += 1

for i in range(len(cast)):
    for j in json.loads(cast[i]):
        cast_score[j['name']] += y_train[i]

for i in cast_score.keys():
    cast_score[i] = cast_score[i]/cast_count[i]

cast_encoded = []
for i in cast:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += cast_score[j['name']]
        cnt += 1
    if cnt != 0:
        cast_encoded.append(tmp/cnt)
    else:
        cast_encoded.append(tmp)
df_xtrain['cast'] = cast_encoded
cast_min = np.min(df_xtrain['cast'])
cast_max = np.max(df_xtrain['cast'])
df_xtrain['cast'] = df_xtrain['cast']-cast_min
df_xtrain['cast'] = ((df_xtrain['cast']/(cast_max-cast_min)))


#crew pre_processing
crew_dep_count = {}
crew_dep_score = {}
crew_job_count = {}
crew_job_score = {}
crew_name_count = {}
crew_name_score = {}
crew = df_xtrain['crew'].tolist()
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
            crew_name_score[j['name']] += y_train[i]

for i in crew:
    for j in json.loads(i):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_name_score[j['name']] = crew_name_score[j['name']]/crew_name_count[j['name']]

crew_encoded = []
for i in crew:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        tmp += crew_name_score[j['name']]
        cnt += 1
    if cnt != 0:
        crew_encoded.append(tmp/cnt)
    else:
        crew_encoded.append(tmp)
df_xtrain['crew'] = crew_encoded
crew_min = np.min(df_xtrain['crew'])
crew_max = np.max(df_xtrain['crew'])
df_xtrain['crew'] = df_xtrain['crew']-crew_min
df_xtrain['crew'] = ((df_xtrain['crew']/(crew_max-crew_min)))

#normalize popularity
popularity_min = np.min(df_xtrain['popularity'])
popularity_max = np.max(df_xtrain['popularity'])
df_xtrain['popularity'] = df_xtrain['popularity']-popularity_min
df_xtrain['popularity'] = (df_xtrain['popularity']/(popularity_max-popularity_min))

#normalize runtime
runtime_min = np.min(df_xtrain['runtime'])
runtime_max = np.max(df_xtrain['runtime'])
df_xtrain['runtime'] = df_xtrain['runtime']-runtime_min
df_xtrain['runtime'] = (df_xtrain['runtime']/(runtime_max-runtime_min))

#normalize vote count
vote_count_min = np.min(df_xtrain['vote_count'])
vote_count_max = np.max(df_xtrain['vote_count'])
df_xtrain['vote_count'] = df_xtrain['vote_count']-vote_count_min
df_xtrain['vote_count'] = (df_xtrain['vote_count']/(vote_count_max-vote_count_min))

#language pre_processing
lang = df_xtrain['original_language']
lang_count = {}
lang_score = {}
for i in lang:
    lang_count[i] = 0
    lang_score[i] = 0
for i in lang:
    lang_count[i] += 1
cnt = 0
for i in lang:
    lang_score[i] += y_train[cnt]
    cnt += 1
for i in lang_count.keys():
    lang_score[i] = lang_score[i]/lang_count[i]
lang_encoded = []
for i in lang:
    lang_encoded.append(lang_score[i])
df_xtrain['original_language'] = lang_encoded

#Normalize Language
lang_min = np.min(df_xtrain['original_language'])
lang_max = np.max(df_xtrain['original_language'])
df_xtrain['original_language'] = df_xtrain['original_language']-lang_min
df_xtrain['original_language'] = (df_xtrain['original_language']/(lang_max-lang_min))

#pre processing dates
release_date = df_xtrain['release_date']
for i in release_date:
    x = i[-4:]
date_count = {}
date_score = {}
for i in release_date:
    x = i[-4:]
    date_count[x] = 0
    date_score[x] = 0
for i in release_date:
    x = i[-4:]
    date_count[x] += 1
cnt = 0
for i in release_date:
    x = i[-4:]
    date_score[x] += y_train[cnt]
    cnt += 1
for i in date_count.keys():
    date_score[i] = date_score[i]/date_count[i]
date_encoded = []
for i in release_date:
    x = i[-4:]
    date_encoded.append(date_score[x])
df_xtrain['release_date'] = date_encoded

#Normalize date
date_min = np.min(df_xtrain['release_date'])
date_max = np.max(df_xtrain['release_date'])
df_xtrain['release_date'] = df_xtrain['release_date']-date_min
df_xtrain['release_date'] = (df_xtrain['release_date']/(date_max-date_min))


#drop columns



#####################################################################################


#genres pre_processing
genres = df_xtest['genres'].tolist()
genres_encoded = []
for i in genres:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in genres_score.keys():
            tmp += genres_score[j['name']]
            cnt += 1
    if cnt != 0:
        genres_encoded.append(tmp/cnt)
    else:
        genres_encoded.append(tmp)
df_xtest['genres'] = genres_encoded
df_xtest['genres'] = df_xtest['genres']-genres_min
df_xtest['genres'] = ((df_xtest['genres']/(genres_max-genres_min)))

#keywords pre_processing
keywords = df_xtest['keywords']
keywords_encoded = []
for i in keywords:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in keywords_score.keys():
            tmp += keywords_score[j['name']]
            cnt += 1
    if cnt != 0:
        keywords_encoded.append(tmp/cnt)
    else:
        keywords_encoded.append(tmp)
df_xtest['keywords'] = keywords_encoded
df_xtest['keywords'] = df_xtest['keywords']-keywords_min
df_xtest['keywords'] = ((df_xtest['keywords']/(keywords_max-keywords_min)))

#production companies pre_processing
prodcomp = df_xtest['production_companies'].tolist()
prodcomp_encoded = []
for i in prodcomp:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in prodcomp_score.keys():
            tmp += prodcomp_score[j['name']]
            cnt += 1
    if cnt != 0:
        prodcomp_encoded.append(tmp/cnt)
    else:
        prodcomp_encoded.append(tmp)
df_xtest['production_companies'] = prodcomp_encoded
df_xtest['production_companies'] = df_xtest['production_companies']-prodcomp_min
df_xtest['production_companies'] = ((df_xtest['production_companies']/(prodcomp_max-prodcomp_min)))

#production countries pre_processing
prodcountry = df_xtest['production_countries'].tolist()
prodcountry_encoded = []
for i in prodcountry:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in prodcountry_score.keys():
            tmp += prodcountry_score[j['name']]
            cnt += 1
    if cnt != 0:
        prodcountry_encoded.append(tmp/cnt)
    else:
        prodcountry_encoded.append(tmp)
df_xtest['production_countries'] = prodcountry_encoded
df_xtest['production_countries'] = df_xtest['production_countries']-prodcountry_min
df_xtest['production_countries'] = ((df_xtest['production_countries']/(prodcountry_max-prodcountry_min)))

#spoken languages pre_processing
spokenlang = df_xtest['spoken_languages'].tolist()
spokenlang_encoded = []
for i in spokenlang:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in spokenlang_score.keys():
            tmp += spokenlang_score[j['name']]
            cnt += 1
    if cnt != 0:
        spokenlang_encoded.append(tmp/cnt)
    else:
        spokenlang_encoded.append(tmp)
df_xtest['spoken_languages'] = spokenlang_encoded
df_xtest['spoken_languages'] = df_xtest['spoken_languages']-spokenlang_min
df_xtest['spoken_languages'] = ((df_xtest['spoken_languages']/(spokenlang_max-spokenlang_min)))

#cast pre_processing
cast = df_xtest['cast'].tolist()
cast_encoded = []
for i in cast:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in cast_score.keys():
            tmp += cast_score[j['name']]
            cnt += 1
    if cnt != 0:
        cast_encoded.append(tmp/cnt)
    else:
        cast_encoded.append(tmp)
df_xtest['cast'] = cast_encoded
df_xtest['cast'] = df_xtest['cast']-cast_min
df_xtest['cast'] = ((df_xtest['cast']/(cast_max-cast_min)))

#crew pre_processing
crew = df_xtest['crew'].tolist()
crew_encoded = []
for i in crew:
    tmp = 0
    cnt = 0
    for j in json.loads(i):
        if j['name'] in crew_name_score.keys():
            tmp += crew_name_score[j['name']]
            cnt += 1
    if cnt != 0:
        crew_encoded.append(tmp/cnt)
    else:
        crew_encoded.append(tmp)
df_xtest['crew'] = crew_encoded
df_xtest['crew'] = df_xtest['crew']-crew_min
df_xtest['crew'] = ((df_xtest['crew']/(crew_max-crew_min)))


#normalize popularity
df_xtest['popularity'] = df_xtest['popularity']-popularity_min
df_xtest['popularity'] = (df_xtest['popularity']/(popularity_max-popularity_min))

#normalize runtime
df_xtest['runtime'] = df_xtest['runtime']-runtime_min
df_xtest['runtime'] = (df_xtest['runtime']/(runtime_max-runtime_min))

#normalize vote count
df_xtest['vote_count'] = df_xtest['vote_count']-vote_count_min
df_xtest['vote_count'] = (df_xtest['vote_count']/(vote_count_max-vote_count_min))

#language pre_processing
lang = df_xtest['original_language'].tolist()
lang_encoded = []
for i in lang:
    if i in lang_score.keys():
        lang_encoded.append(lang_score[i])
    else:
        lang_encoded.append(0)
df_xtest['original_language'] = lang_encoded
#Normalize Language
df_xtest['original_language'] = df_xtest['original_language']-lang_min
df_xtest['original_language'] = (df_xtest['original_language']/(lang_max-lang_min))

#pre processing dates
date_encoded = []
release_date = df_xtest['release_date'].tolist()
for i in release_date:
    x = i[-4:]
    if i in date_score.keys():
        date_encoded.append(date_score[x])
    else:
        date_encoded.append(0)
df_xtest['release_date'] = date_encoded
#Normalize date
df_xtest['release_date'] = df_xtest['release_date']-date_min
df_xtest['release_date'] = (df_xtest['release_date']/(date_max-date_min))


x_train = df_xtrain.iloc[:, :]
x_test = df_xtest.iloc[:, :]

clf = linear_model.LinearRegression()
clf.fit(x_train, y_train)
y_hat = clf.predict(x_test)
print("Linear score " + str(metrics.r2_score(y_test, y_hat)))
print("Linear mse " + str(metrics.mean_squared_error(y_test, y_hat)))

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(x_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
prediction = poly_model.predict(poly_features.fit_transform(x_test))
print("polynomial score " + str(metrics.r2_score(y_test, prediction)))
print("polynomial mse " + str(metrics.mean_squared_error(y_test, prediction)))

model = linear_model.Ridge()
model.fit(x_train, y_train)
y_hat = model.predict(x_test)
ymin = np.min(y_hat)
ymax = np.max(y_hat)
#y_hat = ((y_hat-ymin)*9)/(ymax-ymin))
print(y_hat[:10])
print(y_test[:10])
print("ridge score " + str(metrics.r2_score(y_test, y_hat)))
print("ridge mse " + str(metrics.mean_squared_error(y_test, y_hat)))

model = linear_model.BayesianRidge()
model.fit(x_train, y_train)
y_hat = model.predict(x_test)
print("bayes score " + str(metrics.r2_score(y_test, y_hat)))
print("bayes mse " + str(metrics.mean_squared_error(y_test, y_hat)))
#df.to_csv('out.csv', index=False)
print(df_xtrain.corr())

#corr = df_xtrain.corr(y_train)
#Top 50% Correlation training features with the Value
#top_feature = corr.index[abs(corr['Value']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = df_xtrain[:].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
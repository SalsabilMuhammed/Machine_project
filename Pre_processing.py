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
#data_movies.dropna(how='any', inplace=True)
data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')

df = pd.DataFrame(data_ab)
vote_average = df['vote_average']

#genres pre_processing
genres_count = {}
genres_score = {}
genres = df['genres']

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] = 0
        genres_score[j['name']] = 0

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] += 1

for i in range(len(genres)):
    for j in json.loads(genres[i]):
        genres_score[j['name']] += vote_average[i]

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
df['genres'] = genres_encoded

genres_min = np.min(df['genres'])
genres_max = np.max(df['genres'])
df['genres'] = df['genres']-genres_min
df['genres'] = ((df['genres']/(genres_max-genres_min))*3)



#keywords pre_processing
keywords_count = {}
keywords_score = {}
keywords = df['keywords']
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
df['keywords'] = keywords_encoded
keywords_min = np.min(df['keywords'])
keywords_max = np.max(df['keywords'])
df['keywords'] = df['keywords']-keywords_min
df['keywords'] = ((df['keywords']/(keywords_max-keywords_min))*3)


#production companies pre_processing
prodcomp_count = {}
prodcomp_score = {}
prodcomp = df['production_companies']
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
df['production_companies'] = prodcomp_encoded
prodcomp_min = np.min(df['production_companies'])
prodcomp_max = np.max(df['production_companies'])
df['production_companies'] = df['production_companies']-prodcomp_min
df['production_companies'] = ((df['production_companies']/(prodcomp_max-prodcomp_min))*3)


#production countries pre_processing
prodcountry_count = {}
prodcountry_score = {}
prodcountry = df['production_countries']
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
df['production_countries'] = prodcountry_encoded
prodcountry_min = np.min(df['production_countries'])
prodcountry_max = np.max(df['production_countries'])
df['production_countries'] = df['production_countries']-prodcountry_min
df['production_countries'] = ((df['production_countries']/(prodcountry_max-prodcountry_min))*3)



#spoken languages pre_processing
spokenlang_count = {}
spokenlang_score = {}
spokenlang = df['spoken_languages']
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
df['spoken_languages'] = spokenlang_encoded
spokenlang_min = np.min(df['spoken_languages'])
spokenlang_max = np.max(df['spoken_languages'])
df['spoken_languages'] = df['spoken_languages']-spokenlang_min
df['spoken_languages'] = ((df['spoken_languages']/(spokenlang_max-spokenlang_min))*3)


#cast pre_processing
cast_count = {}
cast_score = {}
cast = df['cast']
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
df['cast'] = cast_encoded
cast_min = np.min(df['cast'])
cast_max = np.max(df['cast'])
df['cast'] = df['cast']-cast_min
df['cast'] = ((df['cast']/(cast_max-cast_min))*3)


#crew pre_processing
crew_dep_count = {}
crew_dep_score = {}
crew_job_count = {}
crew_job_score = {}
crew_name_count = {}
crew_name_score = {}
crew = df['crew']
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
            crew_name_score[j['name']] = crew_name_score[j['name']]/crew_name_count[j['name']]

#for i in crew_name_score.keys():
#     crew_name_score[i] = crew_name_score[i]/crew_name_count[i]

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
df['crew'] = crew_encoded
crew_min = np.min(df['crew'])
crew_max = np.max(df['crew'])
df['crew'] = df['crew']-crew_min
df['crew'] = ((df['crew']/(crew_max-crew_min))*3)

#normalize popularity
popularity_min = np.min(df['popularity'])
popularity_max = np.max(df['popularity'])
df['popularity'] = df['popularity']-popularity_min
df['popularity'] = (df['popularity']/(popularity_max-popularity_min))*3

#normalize runtime
runtime_min = np.min(df['runtime'])
runtime_max = np.max(df['runtime'])
df['runtime'] = df['runtime']-runtime_min
df['runtime'] = (df['runtime']/(runtime_max-runtime_min))*3

#normalize vote count
vote_count_min = np.min(df['vote_count'])
vote_count_max = np.max(df['vote_count'])
df['vote_count'] = df['vote_count']-vote_count_min
df['vote_count'] = (df['vote_count']/(vote_count_max-vote_count_min))*3

#language pre_processing
lang = df['original_language']
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
    lang_score[i] = lang_score[i]/lang_count[i]
lang_encoded = []
for i in lang:
    lang_encoded.append(lang_score[i])
df['original_language'] = lang_encoded
#Normalize Language
lang_min = np.min(df['original_language'])
lang_max = np.max(df['original_language'])
df['original_language'] = df['original_language']-lang_min
df['original_language'] = (df['original_language']/(lang_max-lang_min))*3

#pre processing dates
release_date = df['release_date']
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
    date_score[x] += vote_average[cnt]
    cnt += 1
for i in date_count.keys():
    date_score[i] = date_score[i]/date_count[i]
date_encoded = []
for i in release_date:
    x = i[-4:]
    date_encoded.append(date_score[x])
df['release_date'] = date_encoded
#Normalize date
date_min = np.min(df['release_date'])
date_max = np.max(df['release_date'])
df['release_date'] = df['release_date']-date_min
df['release_date'] = (df['release_date']/(date_max-date_min))


#drop columns
df.drop('homepage', axis=1, inplace=True)
df.drop('original_title', axis=1, inplace=True)
df.drop('overview', axis=1, inplace=True)
df.drop('tagline', axis=1, inplace=True)
df.drop('title_x', axis=1, inplace=True)
df.drop('title_y', axis=1, inplace=True)
df.drop('movie_id', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
df.drop('status', axis=1, inplace=True)

#fill zero budgets and revenues
'''budget = df['budget']
zero_budget = []
z_b = []
revenue = df['revenue']
zero_revenue = []
z_r = []
for i in range(len(budget)):
    tmp = []
    if budget[i] == 0:
        tmp.append(i)
        z_b.append(i)
        tmp.append(vote_average[i])
        zero_budget.append(tmp)
    tmp2 = []
    if revenue[i] == 0:
        tmp2.append(i)
        z_r.append(i)
        tmp2.append(vote_average[i])
        zero_revenue.append(tmp2)

for i in zero_budget:
    zero_budget_sum = 0
    zero_budget_count = 0
    for j in range(len(vote_average)):
        if vote_average[j] == i[1] and j not in z_b:
            zero_budget_count += 1
            zero_budget_sum += budget[j]
    if zero_budget_sum != 0:
        i.append(zero_budget_sum/zero_budget_count)

for i in zero_revenue:
    zero_revenue_sum = 0
    zero_revenue_count = 0
    for j in range(len(vote_average)):
        if vote_average[j] == i[1] and j not in z_r:
            zero_revenue_count += 1
            zero_revenue_sum += budget[j]
    if zero_revenue_sum != 0:
        i.append(zero_revenue_sum/zero_revenue_count)

for i in zero_budget:
    if len(i) > 2:
        budget[i[0]] = i[2]
df['budget'] = budget

for i in zero_revenue:
    if len(i) > 2:
        revenue[i[0]] = i[2]
df['revenue'] = revenue
'''

#normalize budget
df = df[df.budget != 0]
budget_max = np.max(df['budget'])
budget_min = np.min(df['budget'])
dif = budget_max-budget_min
df['budget'] = df['budget']-budget_min
df['budget'] = (df['budget']/dif)*3

revenue_min = np.min(df['revenue'])
revenue_max = np.max(df['revenue'])
df['revenue'] = df['revenue']-revenue_min
df['revenue'] = (df['revenue']/(revenue_max-revenue_min))*3

df.dropna(how='any', inplace=True)

print(df.corr())
X = df.iloc[:, df.keys() != 'vote_average']
print(X.keys())
Y = df.iloc[:, -4]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
print(min(clf.coef_))


y_hat = clf.predict(X_test)
print("Linear score " + str(metrics.r2_score(y_test, y_hat)))
print("Linear mse " + str(metrics.mean_squared_error(y_test, y_hat)))

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print("polynomial score " + str(metrics.r2_score(y_test, prediction)))
print("polynomial mse " + str(metrics.mean_squared_error(y_test, prediction)))

model = linear_model.Ridge()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print("ridge score " + str(metrics.r2_score(y_test, y_hat)))
print("ridge mse " + str(metrics.mean_squared_error(y_test, y_hat)))

model = linear_model.BayesianRidge()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print("bayes score " + str(metrics.r2_score(y_test, y_hat)))
print("bayes mse " + str(metrics.mean_squared_error(y_test, y_hat)))
df.to_csv('out.csv', index=False)

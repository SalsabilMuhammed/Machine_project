import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
#from mlxtend.plotting import plot_linear_regressionfrom
from sklearn.preprocessing import PolynomialFeatures
import json
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from scipy.optimize import fmin_tnc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


import xlrd

#read csv
data_movies = pd.read_csv('tmdb_5000_movies_train.csv')
data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')
df = pd.DataFrame(data_ab)

y = df.iloc[:, df.keys() == 'vote_average']
X = df.iloc[:, df.keys() != 'vote_average']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

df=pd.concat([X_train,y_train],axis=1)


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

new_df = df.copy()

new_df.drop('genres', axis=1, inplace=True)
new_df.drop('keywords', axis=1, inplace=True)
new_df.drop('production_companies', axis=1, inplace=True)
new_df.drop('production_countries', axis=1, inplace=True)
new_df.drop('spoken_languages', axis=1, inplace=True)
new_df.drop('cast', axis=1, inplace=True)
new_df.drop('crew', axis=1, inplace=True)
new_df.drop('original_language', axis=1, inplace=True)
new_df.drop('release_date', axis=1, inplace=True)


col_len = len(new_df['vote_average'])


test_df = pd.concat([X_test,y_test],axis=1)
test_df.drop('homepage', axis=1, inplace=True)
test_df.drop('original_title', axis=1, inplace=True)
test_df.drop('overview', axis=1, inplace=True)
test_df.drop('tagline', axis=1, inplace=True)
test_df.drop('title_x', axis=1, inplace=True)
test_df.drop('title_y', axis=1, inplace=True)
test_df.drop('movie_id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)
test_df.drop('status', axis=1, inplace=True)

newtest_df = test_df.copy()

newtest_df.drop('genres', axis=1, inplace=True)
newtest_df.drop('keywords', axis=1, inplace=True)
newtest_df.drop('production_companies', axis=1, inplace=True)
newtest_df.drop('production_countries', axis=1, inplace=True)
newtest_df.drop('spoken_languages', axis=1, inplace=True)
newtest_df.drop('cast', axis=1, inplace=True)
newtest_df.drop('crew', axis=1, inplace=True)
newtest_df.drop('original_language', axis=1, inplace=True)
newtest_df.drop('release_date', axis=1, inplace=True)

test_col_len = len(newtest_df['vote_average'])

#

#genres pre_processing
genres_count = {}

genres = df['genres'].tolist()

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] = 0

for i in genres:
    for j in json.loads(i):
        genres_count[j['name']] += 1

genres_sorted = {k:v for k,v in sorted(genres_count.items(),key=lambda item:item[1] , reverse=True)}

#print(genres_sorted)

cnt=0
for i in genres_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    #print(i)
    cnt=cnt+1
    new_df[str(i)] = tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#keywords col.................................................

keywords_count = {}

keywords = df['keywords'].tolist()

for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] = 0

for i in keywords:
    for j in json.loads(i):
        keywords_count[j['name']] += 1

keywords_sorted = {k: v for k, v in sorted(keywords_count.items(), key=lambda item: item[1], reverse=True)}
#print(keywords_sorted)

cnt = 0
for i in keywords_sorted.keys():
    if cnt == 100:
        break
    tmp = np.zeros(col_len)
    #print(i)
    cnt = cnt+1
    new_df[str(i)] = tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2


#production_companies col..................................................
production_companies_count = {}
production_companies = df['production_companies'].tolist()

for i in production_companies:
    for j in json.loads(i):
        production_companies_count[j['name']] = 0

for i in production_companies:
    for j in json.loads(i):
        production_companies_count[j['name']] += 1

production_companies_sorted = {k:v for k,v in sorted(production_companies_count.items(),key=lambda item:item[1] , reverse=True)}
#print(production_companies_sorted)

cnt=0
for i in production_companies_sorted.keys():
    if cnt == 50:
        break
    tmp=np.zeros(col_len)
    #print(str(i))
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2



#production_countries col..................................................
production_countries_count = {}

production_countries = df['production_countries'].tolist()

for i in production_countries:
    for j in json.loads(i):
        production_countries_count[j['name']] = 0

for i in production_countries:
    for j in json.loads(i):
        production_countries_count[j['name']] += 1

production_countries_sorted = {k:v for k,v in sorted(production_countries_count.items(),key=lambda item:item[1] , reverse=True)}
#print(production_countries_sorted)

cnt=0
for i in production_countries_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    #print(i)
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#spoken languages col..................................................
spoken_languages_count = {}

spoken_languages = df['spoken_languages'].tolist()

for i in spoken_languages:
    for j in json.loads(i):
        spoken_languages_count[j['name']] = 0

for i in spoken_languages:
    for j in json.loads(i):
        spoken_languages_count[j['name']] += 1

spoken_languages_sorted = {k:v for k,v in sorted(spoken_languages_count.items(),key=lambda item:item[1] , reverse=True)}
#print(spoken_languages_sorted)

cnt=0
for i in spoken_languages_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    #print(i)
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#original_laguage col..................................................
original_language_count = {}
original_language = df['original_language'].tolist()

for i in original_language:
    original_language_count[i] = 0

for i in original_language:
    original_language_count[i] += 1

original_language_sorted = {k:v for k,v in sorted(original_language_count.items(),key=lambda item: item[1], reverse=True)}

cnt=0
for i in original_language_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#cast col..................................................
cast_count = {}

cast = df['cast'].tolist()

for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] = 0

for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] += 1

cast_sorted = {k:v for k,v in sorted(cast_count.items(),key=lambda item: item[1], reverse=True)}
#print(cast_sorted)

cnt=0
for i in cast_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    #print(i)
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#crew col..................................................
crew_count = {}

crew = df['crew'].tolist()

for i in crew:
    for j in json.loads(i):
        crew_count[j['name']] = 0

for i in crew:
    for j in json.loads(i):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_count[j['name']] += 1

crew_sorted = {k:v for k,v in sorted(crew_count.items(),key=lambda item: item[1], reverse=True)}
#print(crew_sorted)

cnt=0
for i in crew_sorted.keys():
    if cnt==100:
        break
    tmp=np.zeros(col_len)
    #print(i)
    cnt=cnt+1
    new_df[str(i)]=tmp
    tmp2 = np.zeros(test_col_len)
    newtest_df[str(i)] = tmp2

#calculate budget average
budget_average = np.mean(df['budget'].tolist())

#calculate revenue average
revenue_average = np.mean(df['revenue'].tolist())

#calculate runtime average
runtime_average = np.mean(df['runtime'].tolist())

#calculate popularity average
popularity_average = np.mean(df['popularity'].tolist())

#calculate vote_count average
vote_count_average = np.mean(df['vote_count'].tolist())

print(new_df.keys())

for i in df.axes[0]:
    for j in json.loads(df.loc[i,'genres']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i,'keywords']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i,'production_companies']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i,'production_countries']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i,'spoken_languages']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i,'crew']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    for j in json.loads(df.loc[i, 'cast']):
        if str(j['name']) in new_df.keys():
            new_df.loc[i, str(j['name'])] = 1

    if df.loc[i, 'original_language'] in new_df.keys():
        new_df.loc[i, df.loc[i, 'original_language']] = 1



for i in test_df.axes[0]:
    for j in json.loads(test_df.loc[i,'genres']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i,'keywords']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i,'production_companies']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i,'production_countries']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i,'spoken_languages']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i,'crew']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    for j in json.loads(test_df.loc[i, 'cast']):
        if str(j['name']) in newtest_df.keys():
            newtest_df.loc[i, str(j['name'])] = 1

    if test_df.loc[i, 'original_language'] in newtest_df.keys():
        newtest_df.loc[i, test_df.loc[i, 'original_language']] = 1

    if str(test_df.loc[i, 'budget']) == 'nan':
        newtest_df.loc[i, 'budget'] = budget_average

    if str(test_df.loc[i, 'revenue']) == 'nan':
        newtest_df.loc[i, 'revenue'] = revenue_average

    if str(test_df.loc[i, 'runtime']) == 'nan':
        newtest_df.loc[i, 'runtime'] = runtime_average

    if str(test_df.loc[i, 'popularity']) == 'nan':
        newtest_df.loc[i, 'popularity'] = popularity_average

    if str(test_df.loc[i, 'vote_count']) == 'nan':
        newtest_df.loc[i, 'vote_count'] = vote_count_average

#NORMALIZATION for training...........................
#normalize budget
budget_max = np.max(new_df['budget'])
budget_min = np.min(new_df['budget'])
dif = budget_max-budget_min
new_df['budget'] = new_df['budget']-budget_min
new_df['budget'] = (new_df['budget']/dif)


#normalize popularity
popularity_min = np.min(new_df['popularity'])
popularity_max = np.max(new_df['popularity'])
new_df['popularity'] = new_df['popularity']-popularity_min
new_df['popularity'] = (new_df['popularity']/(popularity_max-popularity_min))

#revunue
revenue_min = np.min(new_df['revenue'])
revenue_max = np.max(new_df['revenue'])
new_df['revenue'] = new_df['revenue']-revenue_min
new_df['revenue'] = (new_df['revenue']/(revenue_max-revenue_min))

#normalize runtime
runtime_min = np.min(new_df['runtime'])
runtime_max = np.max(new_df['runtime'])
new_df['runtime'] = new_df['runtime']-runtime_min
new_df['runtime'] = (new_df['runtime']/(runtime_max-runtime_min))


#normalize vote count
vote_count_min = np.min(new_df['vote_count'])
vote_count_max = np.max(new_df['vote_count'])
new_df['vote_count'] = new_df['vote_count']-vote_count_min
new_df['vote_count'] = (new_df['vote_count']/(vote_count_max-vote_count_min))



#NORMALIZATION for testing...........................
#normalize budget
newtest_df['budget'] = newtest_df['budget']-budget_min
newtest_df['budget'] = (newtest_df['budget']/dif)


#normalize popularity
newtest_df['popularity'] = newtest_df['popularity']-popularity_min
newtest_df['popularity'] = (newtest_df['popularity']/(popularity_max-popularity_min))

#revunue
newtest_df['revenue'] = newtest_df['revenue']-revenue_min
newtest_df['revenue'] = (newtest_df['revenue']/(revenue_max-revenue_min))

#normalize runtime
newtest_df['runtime'] = newtest_df['runtime']-runtime_min
newtest_df['runtime'] = (newtest_df['runtime']/(runtime_max-runtime_min))


#normalize vote count
newtest_df['vote_count'] = newtest_df['vote_count']-vote_count_min
newtest_df['vote_count'] = (newtest_df['vote_count']/(vote_count_max-vote_count_min))
#normalize vote_average
#normalize vote_average
'''
vote_count_min = np.min(new_df['vote_average'])
vote_count_max = np.max(new_df['vote_average'])
new_df['vote_average'] = new_df['vote_average']-vote_count_min
new_df['vote_average'] = (new_df['vote_average']/(vote_count_max-vote_count_min))*9
'''

Y = new_df.iloc[:, new_df.keys() == 'vote_average']
X = new_df.iloc[:, new_df.keys() != 'vote_average']

Y_test = newtest_df.iloc[:, newtest_df.keys() == 'vote_average']
X_test = newtest_df.iloc[:, newtest_df.keys() != 'vote_average']

linear_clf= linear_model.LinearRegression()
linear_clf.fit(X, Y)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
y_hat = linear_clf.predict(X_test)
print("Linear score " + str(metrics.r2_score(Y_test, y_hat)))
print("Linear mse " + str(metrics.mean_squared_error(Y_test, y_hat)))

ridge_clf = linear_model.Ridge()
ridge_clf.fit(X, Y)
y_hat = ridge_clf.predict(X_test)
print("ridge score " + str(metrics.r2_score(Y_test, y_hat)))
print("ridge mse " + str(metrics.mean_squared_error(Y_test, y_hat)))

bayes_clf = linear_model.BayesianRidge()
bayes_clf.fit(X, Y)
y_hat = bayes_clf.predict(X_test)
print("bayes score " + str(metrics.r2_score(Y_test, y_hat)))
print("bayes mse " + str(metrics.mean_squared_error(Y_test, y_hat)))
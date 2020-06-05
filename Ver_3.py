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
import os
from sklearn.decomposition import PCA
import joblib

def Feature_Scatter(X_test, y_test,cof,bais):
    #print(X_test.keys())
    fig = plt.figure("figure keywords")
    x1 = X_test['keywords'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.plot(x1,(x1*cof[2])+bais, color='green', linewidth=3)
    plt.xlabel=('keywords')
    plt.ylabel=('vote_average')


    fig = plt.figure("figure original_language")
    x1 = X_test['original_language'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('original_language')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[3], color='green', linewidth=3)

    fig = plt.figure("figure popularity")
    x1 = X_test['popularity'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('popularity')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[4], color='green', linewidth=3)

    fig = plt.figure("figure production_companies")
    x1 = X_test['production_companies'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('production_companies')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[5], color='green', linewidth=3)

    fig = plt.figure("figure production_countries")
    x1 = X_test['production_countries'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('production_countries')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[6], color='green', linewidth=3)

    fig = plt.figure("figure revenue")
    x1 = X_test['revenue'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('revenue')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[8], color='green', linewidth=3)

    fig = plt.figure("figure runtime")
    x1 = X_test['runtime'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('runtime')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[9],color='green', linewidth=3)

    fig = plt.figure("figure spoken_languages")
    x1 = X_test['spoken_languages'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('spoken_languages')
    plt.ylabel = ('vote_average')
    plt.plot(x1*cof[10], color='green', linewidth=3)

    fig = plt.figure("figure vote_count")
    x1 = X_test['vote_count'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('vote_count')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[11], color='green', linewidth=3)

    fig = plt.figure("figure cast")
    x1 = X_test['cast'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('cast')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[12], color='green', linewidth=3)

    fig = plt.figure("figure crew")
    x1 = X_test['crew'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('crew')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[13],  color='green', linewidth=3)

    fig = plt.figure("figure genres")
    x1 = X_test['genres'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('genres')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[1], color='green', linewidth=3)

    fig = plt.figure("figure budget")
    x1 = X_test['budget'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('budget')
    plt.ylabel = ('vote_average')
    plt.plot(x1,x1*cof[0], color='green', linewidth=3)

    fig = plt.figure("figure release_date ")
    x1 = X_test['release_date'].tolist()
    x2 = y_test
    plt.scatter(x1, x2)
    plt.xlabel = ('release_date')
    plt.ylabel = ('vote_average')
    plt.plot(x1,(x1*cof[7])+bais, color='green', linewidth=3)

    plt.show()

def estimate_coeff(x,y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    ss_xy = np.sum(y*x) - n*m_y*m_x
    ss_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = ss_xy/ss_xx
    b_0 = m_y - b_1 *m_x
    return [b_0,b_1]

def plot_reg_line(x,y,b):
    plt.scatter(x,y,color= "m",marker="o", s=30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
#read csv
data_movies = pd.read_csv('tmdb_5000_movies_train.csv')
data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')
df = pd.DataFrame(data_ab)

#X_train, X_test, y_train, y_test = train_test_split(principalDf, Y, test_size=0.2)


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


data_movies_test = pd.read_csv('tmdb_5000_movies_testing_regression.csv')
data_credits_test = pd.read_csv('tmdb_5000_credits_test.csv')
test_data = pd.merge(data_movies_test,data_credits_test,  left_on='id', right_on='movie_id')
test_df = pd.DataFrame(test_data)

test_df.drop('homepage', axis=1, inplace=True)
test_df.drop('original_title', axis=1, inplace=True)
test_df.drop('overview', axis=1, inplace=True)
test_df.drop('tagline', axis=1, inplace=True)
test_df.drop('title_x', axis=1, inplace=True)
test_df.drop('title_y', axis=1, inplace=True)
test_df.drop('movie_id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)
test_df.drop('status', axis=1, inplace=True)
test_df.dropna(axis=1,how='all',inplace=True)

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

test_df.dropna(axis=1,how='all',inplace=True)

test_col_len = len(newtest_df['vote_average'])

#print(new_df.loc[0, 'rate'])

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
    if cnt == 80:
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


#print(df)

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
#nemla zerohat w onaat..........................................


#print(df.loc[2656, 'genres'])

#print(len(new_df.axes[0]))
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
    #print(test_df.loc[i, 'budget'])
    if str(test_df.loc[i,'budget']) == 'nan':
        newtest_df.loc[i,'budget'] = budget_average

    if str(test_df.loc[i,'revenue']) == 'nan':
        newtest_df.loc[i,'revenue'] = revenue_average

    if str(test_df.loc[i,'runtime']) == 'nan':
        newtest_df.loc[i,'runtime'] = runtime_average

    if str(test_df.loc[i,'popularity']) == 'nan':
        newtest_df.loc[i,'popularity'] = popularity_average

    if str(test_df.loc[i,'vote_count']) == 'nan':
        newtest_df.loc[i,'vote_count'] = vote_count_average



'''
#NORMALIZATION for training...........................
#normalize budget
budget_max = np.max(new_df['budget'])
budget_min = np.min(new_df['budget'])
dif = budget_max-budget_min
new_df['budget'] = new_df['budget']-budget_min
new_df['budget'] = (new_df['budget']/dif)*9


#normalize popularity
popularity_min = np.min(new_df['popularity'])
popularity_max = np.max(new_df['popularity'])
new_df['popularity'] = new_df['popularity']-popularity_min
new_df['popularity'] = (new_df['popularity']/(popularity_max-popularity_min))*9

#revunue
revenue_min = np.min(new_df['revenue'])
revenue_max = np.max(new_df['revenue'])
new_df['revenue'] = new_df['revenue']-revenue_min
new_df['revenue'] = (new_df['revenue']/(revenue_max-revenue_min))*9

#normalize runtime
runtime_min = np.min(new_df['runtime'])
runtime_max = np.max(new_df['runtime'])
new_df['runtime'] = new_df['runtime']-runtime_min
new_df['runtime'] = (new_df['runtime']/(runtime_max-runtime_min))*9


#normalize vote count
vote_count_min = np.min(new_df['vote_count'])
vote_count_max = np.max(new_df['vote_count'])
new_df['vote_count'] = new_df['vote_count']-vote_count_min
new_df['vote_count'] = (new_df['vote_count']/(vote_count_max-vote_count_min))*9



#NORMALIZATION for testing...........................
#normalize budget
newtest_df['budget'] = newtest_df['budget']-budget_min
newtest_df['budget'] = (newtest_df['budget']/dif)*9


#normalize popularity
newtest_df['popularity'] = newtest_df['popularity']-popularity_min
newtest_df['popularity'] = (newtest_df['popularity']/(popularity_max-popularity_min))*9

#revunue
newtest_df['revenue'] = newtest_df['revenue']-revenue_min
newtest_df['revenue'] = (newtest_df['revenue']/(revenue_max-revenue_min))*9

#normalize runtime
newtest_df['runtime'] = newtest_df['runtime']-runtime_min
newtest_df['runtime'] = (newtest_df['runtime']/(runtime_max-runtime_min))*9


#normalize vote count
newtest_df['vote_count'] = newtest_df['vote_count']-vote_count_min
newtest_df['vote_count'] = (newtest_df['vote_count']/(vote_count_max-vote_count_min))*9
#normalize vote_average
'''
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
if (os.path.exists('linearmodel.pkl')):
    linear_clf = joblib.load('linearmodel.pkl')
else:
    linear_clf.fit(X, Y)
    joblib.dump(linear_clf, 'linearmodel.pkl')
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
y_hat = linear_clf.predict(X_test)
print("Linear score " + str(metrics.r2_score(Y_test, y_hat)))
print("Linear mse " + str(metrics.mean_squared_error(Y_test, y_hat)))
#coef = estimate_coeff(X_test['budget'].tolist(), y_hat.tolist())
#plot_reg_line(X_test['budget'],y_hat,coef)
#Feature_Scatter(X_test,y_hat,linear_clf.coef_,linear_clf.intercept_)
'''
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y)
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print("polynomial score " + str(metrics.r2_score(Y_test, prediction)))
print("polynomial mse " + str(metrics.mean_squared_error(Y_test, prediction)))
'''
ridge_clf = linear_model.Ridge()
if (os.path.exists('ridgemodel.pkl')):
    ridge_clf = joblib.load('ridgemodel.pkl')
else:
    ridge_clf.fit(X, Y)
    joblib.dump(ridge_clf, 'ridgemodel.pkl')
y_hat = ridge_clf.predict(X_test)
print("ridge score " + str(metrics.r2_score(Y_test, y_hat)))
print("ridge mse " + str(metrics.mean_squared_error(Y_test, y_hat)))

bayes_clf = linear_model.BayesianRidge()
if (os.path.exists('bayesmodel.pkl')):
    bayes_clf = joblib.load('bayesmodel.pkl')
else:
    bayes_clf.fit(X, Y)
    joblib.dump(bayes_clf, 'bayesmodel.pkl')
y_hat = bayes_clf.predict(X_test)
print("bayes score " + str(metrics.r2_score(Y_test, y_hat)))
print("bayes mse " + str(metrics.mean_squared_error(Y_test, y_hat)))


'''



#cast pre_processing
cast_count = {}
cast = df_xtrain['cast'].tolist()
for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] = 0

for i in cast:
    for j in json.loads(i):
        cast_count[j['name']] += 1


#crew pre_processing
crew_name_count = {}

crew = df_xtrain['crew'].tolist()
for i in crew:
    for j in json.loads(i):
        crew_name_count[j['name']] = 0

for i in crew:
    for j in json.loads(i):
        if j['job'] == "Director" or j['job'] == "Producer":
            crew_name_count[j['name']] += 1




#normalize popularity
popularity_min = np.min(df_xtrain['popularity'])
popularity_max = np.max(df_xtrain['popularity'])
df_xtrain['popularity'] = df_xtrain['popularity']-popularity_min
df_xtrain['popularity'] = (df_xtrain['popularity']/(popularity_max-popularity_min))*3

#normalize runtime
runtime_min = np.min(df_xtrain['runtime'])
runtime_max = np.max(df_xtrain['runtime'])
df_xtrain['runtime'] = df_xtrain['runtime']-runtime_min
df_xtrain['runtime'] = (df_xtrain['runtime']/(runtime_max-runtime_min))*3

#normalize vote count
vote_count_min = np.min(df_xtrain['vote_count'])
vote_count_max = np.max(df_xtrain['vote_count'])
df_xtrain['vote_count'] = df_xtrain['vote_count']-vote_count_min
df_xtrain['vote_count'] = (df_xtrain['vote_count']/(vote_count_max-vote_count_min))*3

#language pre_processing
lang = df_xtrain['original_language']
lang_count = {}

for i in lang:
    lang_count[i] = 0
for i in lang:
    lang_count[i] += 1
cnt = 0


#Normalize Language
lang_min = np.min(df_xtrain['original_language'])
lang_max = np.max(df_xtrain['original_language'])
df_xtrain['original_language'] = df_xtrain['original_language']-lang_min
df_xtrain['original_language'] = (df_xtrain['original_language']/(lang_max-lang_min))*3

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




data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}


df = pd.DataFrame(data)


address = np.zeros(4)


df['Address'] = address

print(df)
'''
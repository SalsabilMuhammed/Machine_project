import numpy as np
from sklearn.externals import joblib
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
import os
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


import time

#read csv
data_movies = pd.read_csv('tmdb_5000_movies_classification_2.csv')
data_credits = pd.read_csv('tmdb_5000_credits_2.csv')
data_ab = pd.merge(data_movies, data_credits, left_on='id', right_on='movie_id')
df = pd.DataFrame(data_ab)

y = df.iloc[:, df.keys() == 'rate']
X = df.iloc[:, df.keys() != 'rate']
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


col_len = len(new_df['rate'])


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

test_col_len = len(newtest_df['rate'])


#print(new_df.loc[0, 'rate'])

for i in new_df.axes[0]:

    if new_df.loc[i, 'rate'] == 'High' :
        new_df.loc[i, 'rate'] = 2
    elif new_df.loc[i, 'rate'] == 'Intermediate' :
        new_df.loc[i, 'rate'] = 1
    else:
        new_df.loc[i, 'rate'] = 0

for i in newtest_df.axes[0]:
    if newtest_df.loc[i, 'rate'] == 'High' :
        newtest_df.loc[i, 'rate'] = 2
    elif newtest_df.loc[i, 'rate'] == 'Intermediate' :
        newtest_df.loc[i, 'rate'] = 1
    else:
        newtest_df.loc[i, 'rate'] = 0

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

#print(df.loc[2656, 'genres'])

#print(len(new_df.axes[0]))
#print(new_df.keys())

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
'''
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

#Normalization on test data
#normalize budget
dif = budget_max-budget_min
newtest_df['budget'] = newtest_df['budget']-budget_min
newtest_df['budget'] = (newtest_df['budget']/dif)

#normalize popularity
newtest_df['popularity'] = newtest_df['popularity']-popularity_min
newtest_df['popularity'] = (newtest_df['popularity']/(popularity_max-popularity_min))

#revenue
newtest_df['revenue'] = newtest_df['revenue']-revenue_min
newtest_df['revenue'] = (newtest_df['revenue']/(revenue_max-revenue_min))

#normalize runtime
newtest_df['runtime'] = newtest_df['runtime']-runtime_min
newtest_df['runtime'] = (newtest_df['runtime']/(runtime_max-runtime_min))

#normalize vote count
newtest_df['vote_count'] = newtest_df['vote_count']-vote_count_min
newtest_df['vote_count'] = (newtest_df['vote_count']/(vote_count_max-vote_count_min))

'''
#normalize vote_average
'''
vote_count_min = np.min(new_df['vote_average'])
vote_count_max = np.max(new_df['vote_average'])
new_df['vote_average'] = new_df['vote_average']-vote_count_min
new_df['vote_average'] = (new_df['vote_average']/(vote_count_max-vote_count_min))*9
'''

train_time_without_PCA=[]
test_time_without_PCA=[]
accuracy_without_PCA=[]
models=[]
train_time_with_PCA=[]
test_time_with_PCA=[]
accuracy_with_PCA=[]


Y = new_df.iloc[:, new_df.keys() == 'rate']
X = new_df.iloc[:, new_df.keys() != 'rate']

x = StandardScaler().fit_transform(X)

pca = PCA(n_components=150)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


Y_test = newtest_df.iloc[:, newtest_df.keys() == 'rate']
X_test = newtest_df.iloc[:, newtest_df.keys() != 'rate']


tmp_test = StandardScaler().fit_transform(X_test)

pca_test = PCA(n_components=150)
principalComponents_test = pca_test.fit_transform(tmp_test)
principalDf_test = pd.DataFrame(data = principalComponents_test)

'''
svc=SVC(kernel='rbf', C=1)
svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', C=1))
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1,algorithm="SAMME")
model = abc.fit(principalDf, Y)
y_pred = model.predict(principalDf_test)
print("Adaboost svc Accuracy:",metrics.accuracy_score(Y_test, y_pred))
'''



svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', C=1))

if (os.path.exists('svm_PCA.pkl')):
    svm_model_linear_ovr = joblib.load('svm_PCA.pkl')
else:
    s=time.time()
    svm_model_linear_ovr.fit(principalDf, Y)
    e = time.time()
    models.append('svm')
    train_time_with_PCA.append(e - s)
    joblib.dump(svm_model_linear_ovr, 'svm_PCA.pkl')


# model accuracy for X_test
s2=time.time()
accuracy = svm_model_linear_ovr.score(principalDf_test, Y_test)
e2=time.time()
test_time_with_PCA.append(e2-s2)

accuracy_with_PCA.append(accuracy)
print('One VS Rest SVM accuracy: ' + str(accuracy))



scaler = StandardScaler()
scaler.fit(principalDf)

X_train = scaler.transform(principalDf)
X_test = scaler.transform(principalDf_test)


knn = KNeighborsClassifier(n_neighbors=200)

if (os.path.exists('knn_PCA.pkl')):
    knn = joblib.load('knn_PCA.pkl')
else:
    s=time.time()
    knn.fit(X_train, Y)
    e = time.time()
    models.append('knn')
    train_time_with_PCA.append(e - s)
    joblib.dump(knn, 'knn_PCA.pkl')



s2=time.time()
pred_i = knn.predict(X_test)
e2=time.time()
test_time_with_PCA.append(e2-s2)

pred_i=np.expand_dims(pred_i, axis=1)
accuracy = np.mean(pred_i==Y_test)

accuracy_with_PCA.append(float(accuracy))
print("knn"+str(float(accuracy)))


clf_joblib = tree.DecisionTreeClassifier(max_depth=100)
# Load the model from the file
if (os.path.exists('Decisiontree_PCA.pkl')):
    clf_joblib = joblib.load('Decisiontree_PCA.pkl')
else:
    s=time.time()
    clf_joblib.fit(X_train, Y)
    e = time.time()
    models.append('decisiontree')
    train_time_with_PCA.append(e - s)
    joblib.dump(clf_joblib, 'Decisiontree_PCA.pkl')
# Use the loaded model to make predictions


s2=time.time()
y_prediction = clf_joblib.predict(X_test)
e2=time.time()
test_time_with_PCA.append(e2-s2)

y_prediction=np.expand_dims(y_prediction, axis=1)
accuracy = np.mean(y_prediction==Y_test)
accuracy_with_PCA.append(float(accuracy))
print("tree"+str(float(accuracy)))

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME.R",
                         n_estimators=100)

if (os.path.exists('adaboos_PCA.pkl')):
    bdt = joblib.load('adaboos_PCA.pkl')
else:
    s = time.time()
    bdt.fit(X_train, Y)
    e = time.time()
    models.append('Adaboost_tree')
    train_time_with_PCA.append(e - s)
    joblib.dump(clf_joblib, 'adaboos_PCA.pkl')


s2=time.time()
y_prediction = bdt.predict(X_test)
e2=time.time()
test_time_with_PCA.append(e2-s2)

y_prediction=np.expand_dims(y_prediction, axis=1)
accuracy = np.mean(y_prediction==Y_test)
accuracy_with_PCA.append(float(accuracy))
print("The achieved accuracy using Adaboost is " + str(float(accuracy)))

reg = RandomForestClassifier(n_estimators=20, random_state=0)

if (os.path.exists('Random_forst_PCA.pkl')):
    reg = joblib.load('Random_forst_PCA.pkl')
else:
    s = time.time()
    reg.fit(X_train, Y)
    e = time.time()
    models.append('RandomForst')
    train_time_with_PCA.append(e - s)
    joblib.dump(clf_joblib, 'Random_forst_PCA.pkl')


s2=time.time()
y_pred = reg.predict(X_test)
e2=time.time()
test_time_with_PCA.append(e2-s2)
y_pred=np.expand_dims(y_pred, axis=1)
accuracy = np.mean(y_pred==Y_test)
accuracy_with_PCA.append(float(accuracy))
print("random forest " + str(float(accuracy)))


'''
theta=None
def predict_classes(theta,x):
    tmp = np.dot(x, theta)
    tmp2= 1 / (1 + np.exp(-tmp))

    return (tmp2 >= 0.5).astype(int)

def fit(x, y):
    L = 0.01  # The learning Rate
    epochs = 20000  # The number of iterations to perform gradient descent
    numsamples = x.shape[0]
    theta = np.zeros((x.shape[1],1))
    for i in range(epochs):
        tmp = np.dot(x, theta)
        tmp2 = 1 / (1 + np.exp(-tmp))
        djw = (1 / numsamples) * np.dot(x.T, tmp2 - y)
        theta = theta - L * djw
    return theta

X = np.c_[np.ones((principalDf.shape[0], 1)), principalDf]


if (os.path.exists('Logistic_PCA.pkl')):
    theta = joblib.load('Logistic_PCA.pkl')
else:
    s = time.time()
    theta = fit(X, Y)
    e = time.time()
    models.append('logistic')
    train_time_with_PCA.append(e - s)
    joblib.dump(clf_joblib, 'Logistic_PCA.pkl')




Y = np.expand_dims(Y, axis=1)
actual_classes = Y.flatten()
s2=time.time()
predicted_classes = predict_classes(theta,X)
e2=time.time()
test_time_with_PCA.append(e2-s2)

predicted_classes = predicted_classes.flatten()
accuracy = np.mean(predicted_classes == actual_classes)
accuracy_with_PCA.append(float(accuracy))
print(float(accuracy))
'''
print(models)
print(train_time_with_PCA)
print(test_time_with_PCA)
print(accuracy_with_PCA)
def bargraphs(times, model_name, graph_name,label):
    y_position=np.arange(len(model_name))
    plt.bar(y_position,times,align='center',alpha=1)
    plt.xticks(y_position,model_name)
    plt.ylabel(label)
    plt.title(graph_name)
    plt.show()

#bargraphs(train_time_with_PCA,models,"Training time using PCA ",'Time')
#bargraphs(test_time_with_PCA,models,"Testing time using PCA ",'Time')
#bargraphs(accuracy_with_PCA,models,"accuracy using PCA ",'Accuracy')

####################################################################
####################################################################
######### WITHOUT PCA ########################################
####################################################################
####################################################################

Y = new_df.iloc[:, new_df.keys() == 'rate']
X = new_df.iloc[:, new_df.keys() != 'rate']
Y_test = newtest_df.iloc[:, newtest_df.keys() == 'rate']
X_test = newtest_df.iloc[:, newtest_df.keys() != 'rate']



svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', C=1))


if (os.path.exists('svm.pkl')):
    svm_model_linear_ovr = joblib.load('svm.pkl')
else:
    s=time.time()
    svm_model_linear_ovr.fit(X, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(svm_model_linear_ovr, 'svm.pkl')



# model accuracy for X_test
s2=time.time()
accuracy = svm_model_linear_ovr.score(X_test, Y_test)
e2=time.time()
test_time_without_PCA.append(e2-s2)

accuracy_without_PCA.append(accuracy)
print('One VS Rest SVM accuracy: ' + str(accuracy))



scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=200)


if (os.path.exists('knn.pkl')):
    knn = joblib.load('knn.pkl')
else:
    s=time.time()
    knn.fit(X_train, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(knn, 'knn.pkl')



s2=time.time()
pred_i = knn.predict(X_test)
e2=time.time()
test_time_without_PCA.append(e2-s2)

pred_i=np.expand_dims(pred_i, axis=1)
accuracy = np.mean(pred_i==Y_test)

accuracy_without_PCA.append(float(accuracy))
print("knn"+str(float(accuracy)))


D_tree = tree.DecisionTreeClassifier(max_depth=10,min_samples_split=10)
# Load the model from the file
if (os.path.exists('Decision_tree.pkl')):
    D_tree = joblib.load('Decision_tree.pkl')
else:
    s=time.time()
    D_tree.fit(X_train, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(D_tree, 'Decision_tree.pkl')
# Use the loaded model to make predictions


s2=time.time()
y_prediction = D_tree.predict(X_test)
e2=time.time()
test_time_without_PCA.append(e2-s2)

y_prediction=np.expand_dims(y_prediction, axis=1)
accuracy = np.mean(y_prediction==Y_test)
accuracy_without_PCA.append(float(accuracy))
print("tree"+str(float(accuracy)))

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,min_samples_split=10),

                         algorithm="SAMME.R",
                         n_estimators=100)

if (os.path.exists('Adaboost_tree.pkl')):
    bdt = joblib.load('Adaboost_tree.pkl')
else:
    s=time.time()
    bdt.fit(X_train, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(bdt, 'Adaboost_tree.pkl')



s2=time.time()
y_prediction = bdt.predict(X_test)
e2=time.time()
test_time_without_PCA.append(e2-s2)

y_prediction=np.expand_dims(y_prediction, axis=1)
accuracy = np.mean(y_prediction==Y_test)
accuracy_without_PCA.append(float(accuracy))
print("The achieved accuracy using Adaboost is " + str(float(accuracy)))

reg = RandomForestClassifier(n_estimators=20, random_state=0)

if (os.path.exists('Random_forst.pkl')):
    reg = joblib.load('Random_forst.pkl')
else:
    s = time.time()
    reg.fit(X_train, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(reg, 'Random_forst.pkl')


s2=time.time()
y_pred = reg.predict(X_test)
e2=time.time()
test_time_without_PCA.append(e2-s2)
y_pred=np.expand_dims(y_pred, axis=1)
accuracy = np.mean(y_pred==Y_test)
accuracy_without_PCA.append(float(accuracy))
print("random forest " + str(float(accuracy)))



'''
def predict_classes(theta,x):
    tmp = np.dot(x, theta)
    tmp2= 1 / (1 + np.exp(-tmp))

    return (tmp2 >= 0.5).astype(int)

def fit(x, y):
    L = 0.01  # The learning Rate
    epochs = 20000  # The number of iterations to perform gradient descent
    numsamples = x.shape[0]
    theta = np.zeros((x.shape[1],1))
    for i in range(epochs):
        tmp = np.dot(x, theta)
        tmp2 = 1 / (1 + np.exp(-tmp))
        djw = (1 / numsamples) * np.dot(x.T, tmp2 - y)
        theta = theta - L * djw
    return theta

X = np.c_[np.ones((principalDf.shape[0], 1)), principalDf]



if (os.path.exists('logistic.pkl')):
    theta = joblib.load('logistic.pkl')
else:
    theta = fit(X, Y)
    e = time.time()
    train_time_without_PCA.append(e - s)
    joblib.dump(theta, 'logistic.pkl')


#plot decision boundary
x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
theta = theta.flatten()
y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]

Y = np.expand_dims(Y, axis=1)
actual_classes = Y.flatten()
s2=time.time()
predicted_classes = predict_classes(theta,X)
e2=time.time()
test_time_without_PCA.append(e2-s2)

predicted_classes = predicted_classes.flatten()
accuracy = np.mean(predicted_classes == actual_classes)
accuracy_without_PCA.append(float(accuracy))
print(float(accuracy))
'''
print(models)
print(train_time_without_PCA)
print(test_time_without_PCA)
print(accuracy_without_PCA)


def bargraphs(times, model_name, graph_name,label):
    y_position=np.arange(len(model_name))
    plt.bar(y_position,times,align='center',alpha=1)
    plt.xticks(y_position,model_name)
    plt.ylabel(label)
    plt.title(graph_name)
    plt.show()
#bargraphs(train_time_without_PCA,models,"Training time without PCA ",'Time')
#bargraphs(test_time_without_PCA,models,"Testing time without PCA ",'Time')
#bargraphs(accuracy_without_PCA,models,"accuracy without using PCA ",'Accuracy')


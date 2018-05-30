
# coding: utf-8

# In[1]: Reading data


import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Musical_Instruments_5.json.gz')


# In[2]: Distribution plot


import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes=True)

sns.distplot(df.overall);


# In[2]: SVD Algo


from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)



algo = SVD(n_epochs= 10, lr_all= 0.01, reg_all= 0.6)

cross_validate(algo, data, measures=['RMSE', 'MAE'], return_train_measures=True, cv=10, verbose=True)


# In[3]: Top-10 Recommendation


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from sklearn import metrics

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)
trainset = data.build_full_trainset()

algo.fit(trainset)


testset = trainset.build_anti_testset()
predictions = algo.test(testset)

def get_top_n(predictions, n=10):

    
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=10)

op = []

for uid, user_ratings in top_n.items():
    op.append((uid, [iid for (iid, _) in user_ratings]))

print(op)

# In[54]: SVD Model Selection


from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.001, 0.005, 0.01, 0.05, 0.1],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], return_train_measures=True, cv=10)

gs.fit(data)


print(gs.best_score['rmse'])


print(gs.best_params['rmse'])


# In[60]: KNN Model Selection


from surprise import KNNBaseline
from surprise import KNNBasic

param_grid = {'bsl_options': {'method': ['als', 'sgd']},
              'k': range(2,40),
              'sim_options': {'name': ['cosine', 'pearson_baseline'],
                              'user_based': [False]}
              }

gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=10)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# In[4]: SVD Learning Rate Plot


import matplotlib.pyplot as plt
x = [0.001, 0.005, 0.01, 0.05, 0.1]
y = [0.8869, 0.8701, 0.8626, 0.8703, 0.8772]
plt.ylim(0.8, 0.9)
sns.set_style("whitegrid")
plt.xlabel("Learning Rate")
plt.ylabel("Mean RMSE")
sns.pointplot(x=x, y=y) 


# In[75]:

from surprise import KNNBasic 
bsl_options = {'method': 'als',
               'n_epochs': 10,
               }
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBasic(k = 5, bsl_options=bsl_options, sim_options=sim_options)


# In[26]:


bsl_options = {'method': 'sgd', 
               'lr': 0.01
               }
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBasic(k = 5, bsl_options=bsl_options, sim_options=sim_options)


# In[22]: SlopeOne Algo


from surprise import SVD, SlopeOne

algo = SlopeOne()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)


# In[6]: CoClustering


from surprise import CoClustering
from surprise.model_selection import cross_validate

algo = CoClustering(n_cltr_u = 3, n_cltr_i = 3)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)


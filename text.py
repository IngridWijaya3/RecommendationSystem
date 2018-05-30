
# coding: utf-8

# In[19]:


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
    #print(df[i])
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Musical_Instruments_5.json.gz')
#df = getDF('meta_Musical_Instruments.json.gz')


# In[13]:


#df[df['asin']=='B000BRT17K']


# In[2]:


#df.shape
#region = df.related.tolist()
#incoms = [region for region in region if str(region) != 'nan']
#[list(col) for col in zip(*[[d.keys(),d.values()] for d in incoms])]
#it = list(zip([[d.keys(),d.values()] for d in incoms]))
#pd.DataFrame.from_items(it)
#dict(incoms)
#print(df['related'].apply(pd.Series))
#df['title'] = df['title'].replace('\n', '')
#df['description'] = df['description'].replace('\n', '')
#df2 = pd.concat([df['asin'], df.related.apply(pd.Series)], axis=1)
#df.columns
#df3 = df.merge(df2, left_on='asin', right_on='asin', how='left')
#df3.shape
#df3 = df3.drop(['title', 'imUrl', 'description', 0], axis=1)
#df3.to_csv('musicInst_metadata2.csv', index = False)
#df.to_csv('mi_core.csv')


# In[3]:


#df.shape


# In[4]:


#df = df.drop(['reviewerID', 'reviewerName', 'helpful', 'overall', 'summary', 'unixReviewTime', 'reviewTime'], axis=1)


# In[5]:


#df


# In[8]:


# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.spatial.distance import pdist, squareform
#
# titles = df.reviewText
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(titles)
# cs_title = squareform(pdist(X.toarray(), 'cosine'))


# In[22]:


import numpy as np
def sampling_dataset(df):
    count = 5000
    class_df_sampled = pd.DataFrame(columns = ["overall","reviewText"])
    temp = []
    for c in df.overall.unique():
        class_indexes = df[df.overall == c].index
        random_indexes = np.random.choice(class_indexes, count)
        temp.append(df.loc[random_indexes])
        
    for each_df in temp:
        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    
    return class_df_sampled

#df = sampling_dataset(df)
df.reset_index(drop=True,inplace=True)
print (df.head())
print (df.shape)


# In[27]:


from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

lmtzr = WordNetLemmatizer()
w = re.compile("\w+",re.I)

def label_sentences(df):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w,datapoint["reviewText"].lower())
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['SENT_%s' %index]))
    return labeled_sentences

def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2,alpha=0.025, min_alpha=0.025)
    model.build_vocab(labeled_sentences)
    for epoch in range(10):
        model.train(labeled_sentences, epochs=model.epochs, total_examples=model.corpus_count)
        model.alpha -= 0.002 
        model.min_alpha = model.alpha
    
    return model

sen = label_sentences(df)
model = train_doc2vec_model(sen)


# In[29]:


def vectorize_comments(df,d2v_model):
    y = []
    comments = []
    for i in range(0,df.shape[0]):
        label = 'SENT_%s' %i
        comments.append(d2v_model.docvecs[label])
    df['vectorized_comments'] = comments
    
    return df

df = vectorize_comments(df,model)
print (df.sample(5))


# In[31]:


len(df['vectorized_comments'][0])


# In[35]:


from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

def train_classifier(X,y):
    n_estimators = [200,400]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]

    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split}

    clf = GridSearchCV(RFC(verbose=1,n_jobs=1), cv=10, param_grid=parameters)
    clf.fit(X, y)
    return clf

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df["vectorized_comments"].T.tolist(), df["overall"], test_size=0.02, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))

def logistic_predict(train_target, train_regressor):
    logit = sm.Logit(train_target, train_regressor)
    predictor = logit.fit(disp=0)
    print(predictor.summary())
    return predictor


def error_rate_for_model(test_model,
                         train_set, test_set,
                         infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):

    train_target, train_regressor = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressor = sm.add_constant(train_regressor)
    predictor = logistic_predict(train_target, train_regressor)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
                           test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_data]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    print (error_rate, errors, len(test_predictions), predictor)



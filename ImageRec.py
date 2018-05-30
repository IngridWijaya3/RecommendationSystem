
# coding: utf-8

# In[67]:


import pandas as pd
df = pd.read_csv('output_index_36.csv')


# In[68]:


df.columns = ("label", "features")
df.shape


# In[77]:


pdf = pd.eval(df['features'])
#from scipy.stats import pearsonr
#ps = pearsonr(df[7], df[93])


# In[79]:


ps = pearsonr(pdf[7], pdf[28])
ps


# In[63]:


from sklearn.metrics.pairwise import cosine_similarity
df = df.eval(df.features)
cs = cosine_similarity(df)
#cs.shape
#numpy.asmatrix(cs[0])
#cs[7]


# In[64]:


type(cs)
import numpy #93,72, 89
ncs = numpy.sort(cs[7])
ncs[::-1]


# In[33]:



import numpy
xb = pd.eval(df['features'])
xq = pd.eval(df['features'])
xb = numpy.asarray(xb, dtype=numpy.float32)
xq = numpy.asarray(xq, dtype=numpy.float32)
#xb.shape


# In[4]:


import faiss   
import numpy


# In[5]:


index = faiss.IndexFlatL2(100)   

index.add(xb)                  
print(index.ntotal)


# In[ ]:


k = 1               
D, I = index.search(xq, k)     
print (I)


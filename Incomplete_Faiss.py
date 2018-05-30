
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('output.csv')


# In[2]:


df.columns = ("label", "features")


# In[3]:


#xb = df.values
import numpy
xb = pd.eval(df['features'])
xq = pd.eval(df['features'])
xb = numpy.asarray(xb, dtype=numpy.float32)
xq = numpy.asarray(xq, dtype=numpy.float32)
#xb.shape


# In[4]:


xb = numpy.asarray(xb, dtype=numpy.float32).ravel()
xq = numpy.asarray(xq, dtype=numpy.float32).ravel()


# In[5]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(xb[30], xb[35])


# In[6]:


import faiss   
import numpy


# In[5]:


index = faiss.IndexFlatL2(100)   # build the index, d=size of vectors 

# here we assume xb contains a n-by-d numpy matrix of type float32

index.add(xb)                  # add vectors to the index
print(index.ntotal)


# In[ ]:


k = 1               # we want 4 similar vectors
D, I = index.search(xq, k)     # actual search
print (I)


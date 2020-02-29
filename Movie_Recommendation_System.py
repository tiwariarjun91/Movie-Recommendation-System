#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("Desktop/Movie_DataSet.csv")


# In[53]:


df


# In[17]:


features = ['keywords','cast','genres','director']


# In[18]:


def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[19]:


for feature in features:
    df[feature] = df[feature].fillna('') 

#


# In[21]:


df["combined_features"] = df.apply(combine_features,axis=1)


# In[22]:


df["combined_features"]


# In[27]:


df.iloc[4802].combined_features


# In[30]:


cv = CountVectorizer() 
count_matrix = cv.fit_transform(df["combined_features"])


# In[32]:


count_matrix.toarray()


# In[33]:


cosine_sim = cosine_similarity(count_matrix)


# In[34]:


cosine_sim


# In[55]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[61]:


movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[62]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


# In[63]:


i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break


# In[ ]:





# In[ ]:





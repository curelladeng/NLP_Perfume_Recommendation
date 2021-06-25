#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time, os
import sys
import re
import string
import nltk

import pickle
import random

from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

import streamlit as st
import pickle


header = st.beta_container()

with header:
	st.title('Fragrance Recommendation Engine')

with open('product_topic_nmf_all.pickle', 'rb') as read_file:
    product_topic_nmf_all = pickle.load(read_file)
with open('doc_topic.pickle', 'rb') as read_file:
    doc_topics = pickle.load(read_file)

with open('vectorizer.pickle', 'rb') as read_file:
    vec = pickle.load(read_file)
with open('nmf_model.pickle', 'rb') as read_file:
    nmf_model = pickle.load(read_file)


text = st.text_input('Search', '')

def recommender_system(text, product_topic_df, doc_topic, vc, model):
    tx = [text]
    if text in product_topic_df.product_name.tolist():
        idx = pairwise_distances(np.array(product_topic_df.loc[product_topic_df['product_name'] == text][['1','2','3','4','5','6','7']]).reshape(1,-1),doc_topic,metric='cosine').argsort()
        idx = idx[0][0:5].tolist()
        rec =  product_topic_df.loc[idx][['product_name','url']]

    elif text in product_topic_df.brand_name.tolist():
        rec = product_topic_df[product_topic_df['brand_name'] == text].sort_values('review', ascending = False)[['product_name','url']].head(5)
    else:
        vt = vc.transform(tx)
        tt = model.transform(vt)
        idx = pairwise_distances(tt,doc_topic,metric='cosine').argsort()
        idx = idx[0][0:5].tolist()
        rec = product_topic_df.loc[idx][['product_name','url']]
    rec_all = rec['product_name'] +"- " + rec['url']
    return rec_all


button = st.button('Submit')
if button:
    output= recommender_system(text, product_topic_nmf_all, doc_topics, vec, nmf_model)
    st.write(output)


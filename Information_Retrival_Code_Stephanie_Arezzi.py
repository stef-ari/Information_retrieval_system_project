#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import re
import nltk
import glob
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


nltk.download('punkt')
nltk.download('stopwords')
stemming = PorterStemmer()
stop_words = set(stopwords.words('english'))

file_names = glob.glob('data/*.txt')


# In[10]:


def preprocess_text(text):
  '''
    Executes essential preprocessing steps on the document collection including removal of non-alphanumeric characters,
    coverting text to lowercase, tozenising, removal of stop words, and returns as a string

    Parameters:
        text (str): The input text to be preprocessed.

    Returns:
        str: Preprocessed text from documents with non-alphanumeric characters removed, converted to lowercase,
             and filtered for stop words with stemming applied.
    '''
    # remove non-alphanumeric characters, convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    # remove stop words, and step
    tokens = [stemming.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)


# In[ ]:


inverted_index = {}

# Process each document and build the inverted index
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        processed_text = preprocess_text(text)
        words_in_doc = set(processed_text.split())

        # Update the inverted index
        for word in words_in_doc:
            if word not in inverted_index:
                inverted_index[word] = [file_name]
            else:
                inverted_index[word].append(file_name)

#apply tfidf vector and cosine similarity to documents
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([open(file_name, 'r', encoding='utf-8').read() for file_name in file_names])
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[5]:


def query(query_string):
    '''
    Takes a query string and utilises the TF-IDF Vectoriser to return the top 3 relevant documents

    Parameters:
        query (str): The string to be queried. A score is calculated from the documents based on this string.

    Returns:
        Prints the top 3 relevant documents to the query string
    '''
    # intergrate preprocess_text function
    preprocessed_query = preprocess_text(query_string)
    # Calculate the TF-IDF vector for the query
    tfidf = tfidf_vectorizer.transform([preprocessed_query])
    # Calculate cosine similarity between query and all documents
    query_similarity = cosine_similarity(tfidf, tfidf_matrix)[0]
    # Create a list of document and scores
    document_scores = [(file_name, score) for file_name, score in zip(file_names, query_similarity)]
    # Sort the documents by score in descending order
    document_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the top 3 results
    top_3_results = document_scores[:3]
    return top_3_results


# In[ ]:


#input string to query
query_string = input(str)
results = query(query_string)
print(f"Top 3 most relevant documents to the query '{query_string}':")
for rank, (file_name, score) in enumerate(results, start=1):
    print(f"{rank}. Document: {file_name}, Similarity Score: {score}")


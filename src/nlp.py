import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

def quick_clean(text_arr):
    c = [' '.join(row).lower() for row in text_arr.values]
    out=''
    for row in c:
        out+=row
    return out

# compute Term frequency of a specific term in a document
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))

# IDF of a term
def inverseDocumentFrequency(term, documents):
    count = 0
    for doc in documents:
        if term.lower() in doc.lower().split():
            count += 1
    if count > 0:
        return 1.0 + math.log(float(len(documents))/count)
    else:
        return 1.0

# tf-idf of a term in a document
def tf_idf(term, document, documents):
    tf = termFrequency(term, document)
    idf = inverseDocumentFrequency(term, documents)
    return tf*idf

# make tfidf bow, use collocations=False in word cloud
def tfidf_bow(corpus, max_df=.95, min_df=2, max_features=1000):
    count_vec = CountVectorizer(max_df=max_df, 
                         min_df=min_df,
                         max_features=max_features,
                         stop_words='english')
    cv = count_vec.fit_transform(corpus)
    cv_terms = count_vec.get_feature_names()
    term_counts = cv.toarray().sum(axis=0)
    d = dict(zip(cv_terms, term_counts))
    
    d0 = {}
    stemmer = SnowballStemmer("english")
    for k, v in d.items():
        k = stemmer.stem(k)
        d0[k]=v
        
    d1 = {}
    for k, v in d0.items():
        k = ''.join([char for char in k if char not in '0123456789'])
        if len(k)>2:
            d1[k] = ' '.join([k]*v)
    bow = ''.join(d1.values())
    
    return bow
    
    
def word_cloud(text, collocations=False):
    wordcloud = WordCloud(stopwords=STOPWORDS, 
                        width=800, 
                        height=400, 
                        min_word_length=3,
                        max_words=100, 
                        collocations=collocations).generate(text)
    return wordcloud


if __name__=="__main__":
    
    all_the_data_train = pd.read_csv('../data/all_the_data_train.csv')
    all_the_data_holdout = pd.read_csv('../data/all_the_data_holdout.csv')
    all_df = pd.concat([all_the_data_train, all_the_data_holdout])

    sc_nlp = all_df[['all']][all_df.pcd_mo_year>='2020-03'] 
    s19_nlp = all_df[['all']][(all_df.pcd_mo_year<='2019-12') & (all_df.pcd_mo_year>'2019-03')] 

    covid = quick_clean(sc_nlp)
    pre_2019 = quick_clean(s19_nlp)
    
    pre_2019_plot = word_cloud(pre_2019)
    fig, ax = plt.subplots()
    ax.imshow(pre_2019_plot, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('March-December 2019')
    plt.show()
    
    covid_plot = word_cloud(covid)
    fig, ax = plt.subplots()
    ax.imshow(covid_plot, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('March-December 2020')
    plt.show()
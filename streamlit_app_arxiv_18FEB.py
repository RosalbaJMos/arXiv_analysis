import streamlit as st
import sys
import base64
import io
import logging
import re
import warnings
import webbrowser
from pprint import pprint
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
import string
punct=string.punctuation

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

#sklearn 
from sklearn import datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## plot
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.colors as mcolors
from wordcloud import WordCloud


# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import en_core_web_sm 
nlp = en_core_web_sm.load()


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'due', 'edu', 'observe', 'find', 'set', 'cal', 'solve', 'example', 'text', 
                    'provide', 'increase', 'decrease', 'context', 'achieve', 'large', 'number', 'step', 'give', 
                    'equation', 'solution', 'obtain', 'non', 'paper', 'consider', 'propose', 'important', 'approach',
                     'high', 'low', 'first', 'discuss', 'system', 'term', 'problem', 'model', 'various', 'general', 'require', 'property', 
                     'demonstrate', 'car', 'infer', 'review', 'identify', 'result', 'results', 'new', 'research', 
                     'parameter',  'however', 'study', 'show', 'method', 'field', 'use', 'not', 'would', 'say', 'thus', 
                     'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 
                     'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 
                     'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'work', 'well', 'small', 'datum'])

data_dict = {'Physics' : 'Phys_last_200Pubs_2021.csv',
            'Biology' : 'BIO_last_200Pubs_2021.csv',
            'CS' : 'CS_last_200Pubs_2021.csv',
            'Math': 'Math_last_200Pubs_2021.csv'}


fields = ['Physics','Biology','Computer Science']


st.sidebar.image('arXiv.png')
st.title("The hottest scientific topics on arXiv in 2021 ")
st.subheader('*This is an App that analyzes the abstracts of articles published on arXiv in 2021 and classifies your own abstract*')

url = 'https://github.com/RosalbaJMos/arXiv_Publications_Analysis'

if st.button('My GitHub'):
    webbrowser.open_new_tab(url)


############################################################
# MY FUNCTIONS TO DO PRE-PROCESSING AND TO BUILD THE PLOTS
############################################################
# 1. Funtions to Download the produced figures
def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

# 2. Functions to clean:
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) ##Tokenize 
        yield(sent) 

# 3. Function to tokenize and do the lematization and NLP data preprocessing: 
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords and do the Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm")
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

# 4. Plot Word Count and Weights of Topic Keywords
def plot_topics(topics, data_abstract_ready):      
    data_flat = [w for w_list in data_abstract_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])  

    fig, axes = plt.subplots(3, 2, figsize=(16,16), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.05); ax.set_ylim(0, 500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    st.pyplot(fig)
    st.markdown(imagedownload(plt,'plot_top_words_per_topic.pdf'), unsafe_allow_html=True)


# 5. Wordcloud of Top N words in each topic
def wordclouds(topics):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] 

    cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(16,16), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot_top_words_per_topic.pdf'), unsafe_allow_html=True)

# 6. Another function to clean and tokenize my text 
def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens=[]
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens

# 7. Function to plot the number of articles per month
def articles(Publications2021):
    time=Publications2021['Month']
    phys=Publications2021['pubs_PHY']
    math=Publications2021['pubs_MATH']
    cs=Publications2021['pubs_CS']
    bio=Publications2021['pubs_BIO']
    
    fig = plt.figure(figsize=(12, 8))

    # Customizing the axes
    plt.xticks(rotation=45, color='teal', size=12)
    plt.yticks(rotation=45, color='teal', size=12)

    # plot lines
    plt.plot(time, phys, label = "Physics",  marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(time, math , label = "Math",  marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(time, cs, label = "CS",  marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(time, bio, label = "BIOLOGY",  marker='o', linestyle='dashed', linewidth=2, markersize=12)

    plt.xlabel('Month', size=16)
    plt.ylabel('Number of articles', size=16)

    plt.legend()
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot_publications_in_2021.pdf'), unsafe_allow_html=True)

##################################
# MY MAIN (main) FUNCTION
##################################
#@st.cache(suppress_st_warning=True)
def main():

    st.markdown('** :small_orange_diamond:  1.0 This plot shows the number of articles published per month on arXiv in 2021 corresponding to the fields of Physics, Maths, Computer Science, and Biology**')
    Publications2021 = pd.read_csv('Pubs_PhyMatCSBIO2021.csv') 
    articles(Publications2021)

    data_key = st.sidebar.selectbox('2.1. Select the field', data_dict.keys())
    
    try:
        df = pd.read_csv(data_dict[data_key], parse_dates=['abstract'])
        st.markdown('** :small_orange_diamond: 2.0. This is the dataframe of ' + data_key + '. If you wish to analyze the topics of another field, please select the field on the sidebar at the left :point_left:**')
        st.write(df.head(10))

        #pre-processing our text
        data_abstract = df.abstract.values.tolist()
        data_abstract_words = list(sent_to_words(data_abstract))
        data_abstract_ready = process_words(data_abstract_words)

        # Create Dictionary
        id2word = corpora.Dictionary(data_abstract_ready)
        # Create Corpus
        corpus = [id2word.doc2bow(text) for text in data_abstract_ready]
        #Build our Topic (LDA model)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)
        
        ## call my plot_topics function
        topics = lda_model.show_topics(formatted=False)
        st.markdown('**2.2. These are the keywords that appear more often in the field of ' + data_key + ' when dividing it into six main topics**')
        plot_topics(topics, data_abstract_ready)

        ## call my wordclouds function
        wordclouds(topics)
    except FileNotFoundError as e:
        st.error("Le fichier n'existe pas ! Veuillez recommencer avec un autre fichier")
    except Exception as e:
        st.error(sys.exc_info()[0])
        st.error(e)

main()
    ##################################  HERE STARTS THE PART OF ML ######################################################
def my_predicting_machine(Abstract):
    st.write('Your abstract', Abstract)

    df = pd.read_csv('Sample_PhBiCS.csv') 
    X=df['abstract']
    y=df['Field']
    
    # Clean, tokenize, and buld the vectorization from my text by using the function "text_data_cleaning" 
    tfidf= TfidfVectorizer(tokenizer=text_data_cleaning)
    #use the LinearSVC classifier
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm) 
    # Split my variable (text) and my target (field)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    # Use a Pipeline 
    clf=Pipeline([('tfidf', tfidf), ('clf',clf)])
    # Fit my machine 
    clf.fit(X_train, y_train)
    # Using my Machine to do Predictions on the INPUT ABSTRACT
    X_abstract=text_data_cleaning(Abstract)
    X_abstract_str=" ".join(str(x) for x in X_abstract)
    X_abstract_str=[X_abstract_str]
    Field_Predicted=clf.predict(X_abstract_str)
    Probabilities=clf.predict_proba(X_abstract_str)

    field = fields[Field_Predicted[0]]
    df_fields = pd.DataFrame (fields, columns = ['Field'])
    df_fields['Percentage']=(Probabilities[0]*100)

    st.write('** :white_check_mark:  Your abstract belongs to the field of :**', field)
    st.write('**This is the percent accuracy match of your abstract when classifying it into the three main fields.  :**', df_fields)

st.markdown('**:small_orange_diamond: 3.0. In this part of the App, you can enter your abstract to gain insight on its corresponding category (Field)**')
Abstract = st.text_input('Enter your abstract', ' ')

if st.button('Press to classify your abstract'):
    #st.cache(suppress_st_warning=True)
    my_predicting_machine(Abstract)


# Have to pip3 install <packages> on local machine before import.
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from string import punctuation
from string import digits
from nltk.corpus import wordnet
from sklearn.feature_extraction import text, stop_words

#######################################################
#### Define Custom preprocessor for CountVectorizer ###
#######################################################

def my_custom_preprocessor(doc_string):
    # do all data preprocessing here
    
    # Lower case
    doc_string=doc_string.lower()
    
    # Remove Numbers
    remove_digits = str.maketrans('', '', digits)
    doc_string.translate(remove_digits)
    
    # Convert to tokenized form....
    tokens = nltk.tokenize.word_tokenize(doc_string)
    # Iterate through list of tokens (words) and remove all numbers
    tokens = [word for word in tokens if word.isalpha()]
    # Iterate through list of tokens (words) and stem (shorten) each word
    port_stemmer = PorterStemmer()
    tokens = [port_stemmer.stem(words) for words in tokens ]
    
    ###############################
    #### Lemmatize with pos_tag ###
    ###############################
    
    lemmatizer = WordNetLemmatizer()
    
    # Convert between two different tagging schemes
    def change_tags(penntag):
        morphy_tag = {'NN':'n', 'JJ':'a',
                      'VB':'v', 'RB':'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n'
        
    tokens = [lemmatizer.lemmatize(word.lower(), pos=change_tags(tag)) for word, tag in pos_tag(tokens)]
    
    # Rejoin List of tokens and return that single document-string
    return ' '.join(tokens)


#####################################################
#### Define Custom stop words for CountVectorizer ###
#####################################################

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

# Run stop_words through the same pre-processor as the document-matrix
# This will apply stemmed/lemmatized stop_woirds to stemmed/lemmatized tokenized document lists
def process_stop_words(stop_word_set):
    doc_string = ' '.join(stop_word_set)
    return my_custom_preprocessor(doc_string).split()

#############################
#### Problem #3 Execution ###
#############################

# Problem Setup/Definition:
import numpy as np
np.random.seed(42)
import random
random.seed(42)

# Only take a specific selection (8) of the 20 available categories
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']

# Load a training data sets consisting of those 8 categories
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
print("\n\n" + '-'*40 + "\n\n")

# Define the CountVectorizer = document-term matrix, after stemming + lemmatizing
train_count_vectorizer = CountVectorizer(min_df=3, preprocessor=my_custom_preprocessor, stop_words=process_stop_words(combined_stopwords))
train_doc_term_matrix = train_count_vectorizer.fit_transform(train_dataset.data)
print("Settings of the CountVectorizer(): " + str(train_count_vectorizer))
print("\n\n" + '-'*40 + "\n\n")

# Start TD-DIF Transform process
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_doc_term_matrix)

print("Number of articles within the TRAIN Dataset: " + str(len(train_dataset.filenames)))
print("Number of Features (unique words) in TRAINING dataset (After Processing): "+ str(len(train_count_vectorizer.get_feature_names())))
print("Shape of TRAINING document-count-matrix: " + str(train_doc_term_matrix.shape))
print("Shape of TRAINING TF-IDF Matrix: " + str(train_tfidf.shape))
print("\n\n" + '-'*40 + "\n\n")

#####################
#### LSI Analysis ###
#####################
from sklearn.decomposition import TruncatedSVD

svd_settings = TruncatedSVD(n_components=50, random_state=0)
reduced_LSI_train_matrix = svd_settings.fit_transform(train_tfidf)

print("Shape of tf-idf matrix after LSI reduction (Top 50 words): " + str(reduced_LSI_train_matrix.shape))
print("\n\n" + '-'*40 + "\n\n")

#####################
#### NMF Analysis ###
#####################
from sklearn.decomposition import NMF

nmf_settings = NMF(n_components=50, init='random', random_state=0)
reduced_NMF_train_matrix = nmf_settings.fit_transform(train_tfidf)
nmf_settings_components = nmf_settings.components_

print("Shape of tf-idf matrix after NMF reduction (Top 50 words): "+str(reduced_NMF_train_matrix.shape))
print("\n\n" + '-'*40 + "\n\n")

# Calculate LSI/NMF Values: All performed on train_dataset, save test_dataset for prediction/k-fold analysis
print("Calculated LSI value:")
print(np.sum(np.array(train_tfidf - reduced_LSI_train_matrix.dot(svd_settings.components_)) ** 2))
print("Calculated NMF value")
print(np.sum(np.array(train_tfidf - reduced_NMF_train_matrix.dot(nmf_settings_components)) **2))



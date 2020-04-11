from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF


pipeline1 = Pipeline([
    ('CountVectorizer', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim_NMF', NMF(n_components=50, init='random', random_state=0)),
    ('Log_Reg', MultinomialNB()),
])
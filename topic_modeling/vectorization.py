import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def count_vectorize(df):
    """
    Vectorizes the input data using CountVectorizer.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing a column 'cleaned_statement' with text data.

    Returns:
    tuple: A tuple containing the tokenized data and the CountVectorizer instance.
    """
    tokeniser = CountVectorizer(max_df=0.95, min_df=0.02)
    tokenised_data = tokeniser.fit_transform(df['cleaned_statement'])

    joblib.dump(tokeniser, '../models/count_vectorizer.pkl')
    return tokenised_data, tokeniser


def tfidf_vectorize(df):
    """
    Vectorizes the input data using TfidfVectorizer.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing a column 'cleaned_statement' with text data.

    Returns:
    tuple: A tuple containing the TF-IDF vectorized data and the TfidfVectorizer instance.
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.02)
    tfidf_vectorized = tfidf_vectorizer.fit_transform(df['cleaned_statement'])

    print(f'The number of features/tokens after TFIDF Vectorizer is {len(tfidf_vectorizer.get_feature_names_out())}\n')

    joblib.dump(tfidf_vectorizer, '../models/tfidf_vectorizer.pkl')
    return tfidf_vectorized, tfidf_vectorizer


def load_vectorizer(tfidf=False):
    file_name = 'tfidf_vectorizer.pkl' if tfidf else 'count_vectorizer.pkl'
    return joblib.load(f"../models/{file_name}")


def encode_bm25(df):
    """
    Encodes the input data using the BM25 algorithm.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing a column 'cleaned_statement' with text data.

    Returns:
    tuple: A tuple containing the document-term matrix (dtm) and the vocabulary.
    """
    tokenized_docs = [doc.lower().split() for doc in df['cleaned_statement']]

    bm25Okapi = BM25Okapi(tokenized_docs)

    vocabulary = list(set(word for doc in tokenized_docs for word in doc))
    dtm = np.zeros((len(df['cleaned_statement']), len(vocabulary)))
    for doc_index, doc in enumerate(tokenized_docs):
        for word_index, word in enumerate(vocabulary):
            dtm[doc_index, word_index] = bm25Okapi.get_scores([word])[doc_index]

    joblib.dump({"vocabulary": vocabulary, "dtm": dtm}, "../models/bm25_data.pkl")
    return dtm, vocabulary


def load_bm25():
    """
    Loads the BM25 data from a pickle file.

    Returns:
    tuple: A tuple containing the document-term matrix (dtm) and the vocabulary.
    """
    data = joblib.load("../models/bm25_data.pkl")
    vocabulary = data["vocabulary"]
    dtm = data["dtm"]
    return dtm, vocabulary

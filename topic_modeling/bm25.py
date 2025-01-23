import numpy as np
import pyLDAvis
import torch
from rank_bm25 import BM25Okapi

from validate import plot_wordcloud


def convert_data(data, save_path= "../models/bm25_data.pt"):
    """
    Converts the input data into a document-term matrix (DTM) using the BM25 algorithm and saves it to a file.

    Args:
        data (pd.DataFrame): A dictionary containing the cleaned statements under the key 'cleaned_statement'.
        save_path (str): The path where the resulting DTM and vocabulary will be saved. Default is './models/bm25_data.pt'.

    Returns:
        tuple: A tuple containing the DTM as a numpy array and the vocabulary as a list of words.
    """
    tokenized_docs = [doc.lower().split() for doc in data['cleaned_statement']]

    bm25 = BM25Okapi(tokenized_docs)

    vocabulary = list(set(word for doc in tokenized_docs for word in doc))

    num_docs = len(data['cleaned_statement'])
    num_words = len(vocabulary)
    dtm = torch.zeros((num_docs, num_words), device="cuda")  # GPU Tensor

    for word_index, word in enumerate(vocabulary):
        word_scores = torch.tensor(bm25.get_scores([word]), device="cuda")
        dtm[:, word_index] = word_scores

    dtm = dtm.cpu().numpy()
    torch.save({"vocabulary": vocabulary, "dtm": dtm}, save_path)

    return dtm, vocabulary


def load_bm25_torch(save_path="bm25_data.pt"):
    """
    Loads the BM25 document-term matrix (DTM) and vocabulary from a file.

    Args:
        save_path (str): The path to the file where the DTM and vocabulary are saved. Default is 'bm25_data.pt'.

    Returns:
        tuple: A tuple containing the DTM as a numpy array and the vocabulary as a list of words.
    """
    data = torch.load(f"../models/{save_path}")
    vocabulary = data["vocabulary"]
    dtm = data["dtm"]
    return dtm, vocabulary


def visualise(lda_model, dtm_sparse, vocabulary):
    """
    Visualizes the LDA model using pyLDAvis.

    Args:
        lda_model: The trained LDA model.
        dtm_sparse: The document-term matrix in sparse format.
        vocabulary: The list of words in the vocabulary.
    """
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    doc_topic_dists = lda_model.transform(dtm_sparse)
    doc_lengths = dtm_sparse.sum(axis=1).A1
    term_freqs = dtm_sparse.sum(axis=0).A1
    vocab = vocabulary

    vis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_freqs,
        sort_topics=False
    )

    pyLDAvis.save_html(vis_data, "../static/html/lda_visualisation_bm25.html")


def top_word(model, vocabulary):
    """
    Generates and plots word clouds for the top words in each topic of the LDA model.

    Args:
        model: The trained LDA model.
        vocabulary: The list of words in the vocabulary.
    """
    top_words = top_n_terms(vocabulary, model, 50)
    for topic in top_words:
        plot_wordcloud(topic)


def top_n_terms(vocabulary, model, n_top_words):
    """
    Extracts the top N terms for each topic from the LDA model.

    Args:
        vocabulary (list): The list of words in the vocabulary.
        model: The trained LDA model.
        n_top_words (int): The number of top words to extract for each topic.

    Returns:
        list: A list of dictionaries where each dictionary contains the top words and their weights for a topic.
    """
    top_words_topic = []
    for topic, weights in enumerate(model.components_):
        word_weights = dict()

        indices = np.argsort(weights)[::-1][:n_top_words]
        top_words = [vocabulary[i] for i in indices]
        top_weights = weights[indices]

        for i in range(len(top_words)):
            word_weights[top_words[i]] = top_weights[i]
        top_words_topic.append(word_weights)
    return top_words_topic

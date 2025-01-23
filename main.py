from scipy.sparse import csr_matrix

import LDA
import bm25
from explor_data import *
from type import ModelType
from vectorization import *

if __name__ == "__main__":
    data = load()
    data_df = convert_to_pandas(data)

    # Exploratory Data Analysis
    print(data)
    show_data_info(data_df)

    text_length_distribution(data_df)
    issue_type_distribution(data_df)
    text_length_distribution_by_issue_type(data_df)

    cleaned_data = clean_date(data_df)
    common_words(cleaned_data)
    common_words_by_issue_type(cleaned_data)

    # LDA
    tokenised_data, tokeniser = count_vectorize(cleaned_data)
    lda_model = LDA.fit(tokenised_data)
    best_lda_model = LDA.get_and_save_best(lda_model, tokenised_data)

    LDA.visualise(best_lda_model, tokenised_data, tokeniser)
    LDA.evaluate(cleaned_data, tokenised_data, best_lda_model, ModelType.LDA)
    LDA.top_word(tokeniser, best_lda_model)

    # TFIDF
    tfidf_vectorized, tfidf_vectorizer = tfidf_vectorize(cleaned_data)
    lda_model_2 = LDA.fit(tfidf_vectorized)
    best_lda_model_tfidf = LDA.get_and_save_best(lda_model_2, tfidf_vectorized, ModelType.TFIDF)

    LDA.visualise(best_lda_model_tfidf, tfidf_vectorized, tfidf_vectorizer, ModelType.TFIDF)
    LDA.evaluate(cleaned_data, tfidf_vectorized, best_lda_model_tfidf, ModelType.TFIDF)
    LDA.top_word(tfidf_vectorizer, best_lda_model_tfidf)

    # BM25
    bm25_data, vocabulary = bm25.fit(cleaned_data)
    dtm_sparse = csr_matrix(bm25_data)
    lda_model_3 = LDA.fit(dtm_sparse)

    best_lda_model_bm25 = LDA.get_and_save_best(lda_model_3, dtm_sparse)
    bm25.visualise(best_lda_model_bm25, dtm_sparse, vocabulary)
    LDA.evaluate(cleaned_data, dtm_sparse, best_lda_model_bm25, ModelType.BM)
    bm25.top_word(best_lda_model_bm25, vocabulary)

from enum import Enum


class ModelType(Enum):
    """
    Enum representing different model types.

    Attributes:
        LDA (tuple): Represents the LDA model with its associated file and description.
        TFIDF (tuple): Represents the TF-IDF model with its associated file and description.
        BM (tuple): Represents the BM25 model with its associated file and description.
    """
    LDA = ("lda", "best_lda_model.pkl", " ")
    TFIDF = ("tfidf", "best_lda_model_tfidf.pkl", "with TF-IDF ")
    BM = ("bm", "best_lda_model_bm25.pkl", "with BM25 ")

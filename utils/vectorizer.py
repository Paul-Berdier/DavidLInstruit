import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextVectorizer:
    """
    Classe utilitaire pour vectoriser des textes avec TF-IDF et Word2Vec.
    """

    def __init__(self, method="tfidf"):
        self.method = method
        self.vectorizer = None
        self.model = None
        logging.info(f"Initialisation du vectoriseur avec méthode : {self.method}")

    def fit_transform_tfidf(self, corpus: List[str]):
        """
        Entraîne un TF-IDF sur le corpus et retourne la matrice.
        """
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logging.info("TF-IDF entraîné sur le corpus.")
        return tfidf_matrix

    def transform_tfidf(self, texts: List[str]):
        """
        Applique la vectorisation TF-IDF sur des nouveaux textes.
        """
        if self.vectorizer is None:
            raise ValueError("Le TF-IDF n’a pas encore été entraîné.")
        return self.vectorizer.transform(texts)

    def train_word2vec(self, tokenized_corpus: List[List[str]], vector_size=100, window=5, min_count=2):
        """
        Entraîne un modèle Word2Vec sur un corpus déjà tokenisé.
        """
        self.model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count)
        logging.info("Modèle Word2Vec entraîné.")

    def get_word_vector(self, word: str):
        """
        Retourne le vecteur d’un mot donné.
        """
        if self.model is None:
            raise ValueError("Le modèle Word2Vec n’a pas encore été entraîné.")
        return self.model.wv[word] if word in self.model.wv else None

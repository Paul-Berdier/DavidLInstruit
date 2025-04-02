import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from typing import List, Optional
import joblib
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TextVectorizer:
    """
    Classe pour vectoriser des textes avec TF-IDF ou Word2Vec.
    """

    def __init__(self, method="tfidf"):
        self.method = method
        self.vectorizer = None
        self.model = None
        logging.info(f"Initialisation du vectoriseur avec méthode : {self.method}")

    # ---------- TF-IDF ----------
    def fit_transform_tfidf(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(corpus)
        logging.info("TF-IDF entraîné sur le corpus.")
        return X

    def transform_tfidf(self, texts: List[str]):
        if self.vectorizer is None:
            raise ValueError("Le TF-IDF n’a pas encore été entraîné.")
        return self.vectorizer.transform(texts)

    # ---------- Word2Vec ----------
    def train_word2vec(self, tokenized_corpus: List[List[str]], vector_size=100, window=5, min_count=2):
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count
        )
        logging.info("Modèle Word2Vec entraîné.")

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        if self.model is None:
            raise ValueError("Le modèle Word2Vec n’a pas encore été entraîné.")
        return self.model.wv[word] if word in self.model.wv else None

    def transform_word2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Transforme un corpus tokenisé en une matrice d’embeddings (moyenne des vecteurs de mots).
        """
        if self.model is None:
            raise ValueError("Le modèle Word2Vec n’a pas encore été entraîné.")

        embeddings = []
        for tokens in tokenized_texts:
            word_vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if word_vecs:
                avg_vec = np.mean(word_vecs, axis=0)
            else:
                avg_vec = np.zeros(self.model.vector_size)
            embeddings.append(avg_vec)
        return np.array(embeddings)

    # ---------- Sauvegarde ----------
    def save(self, path: str):
        if self.method == "tfidf" and self.vectorizer:
            joblib.dump(self.vectorizer, path)
            logging.info(f"TF-IDF sauvegardé à {path}")
        elif self.method == "word2vec" and self.model:
            self.model.save(path)
            logging.info(f"Word2Vec sauvegardé à {path}")
        else:
            raise ValueError("Aucun modèle à sauvegarder ou méthode inconnue.")

    def load(self, path: str):
        if self.method == "tfidf":
            self.vectorizer = joblib.load(path)
            logging.info(f"TF-IDF chargé depuis {path}")
        elif self.method == "word2vec":
            self.model = Word2Vec.load(path)
            logging.info(f"Word2Vec chargé depuis {path}")
        else:
            raise ValueError("Méthode de vectorisation non supportée.")

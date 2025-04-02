import logging
from typing import Tuple
import pandas as pd
from utils.preprocessing import TextPreprocessor
from utils.vectorizer import TextVectorizer
from models.ml.ml_classifier import MLTextClassifier
from models.dl.dl_classifier import DLTextClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_contextual_model(df: pd.DataFrame, model_type: str = "ml") -> Tuple[object, TextVectorizer]:
    """
    Entraîne un modèle contextuel supervisé à la volée (ML ou DL).

    :param df: DataFrame contenant les colonnes 'text' et 'label'
    :param model_type: 'ml' (TF-IDF + LogisticRegression) ou 'dl' (Word2Vec + LSTM)
    :return: tuple (modèle entraîné, encodeur de texte utilisé)
    """
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir les colonnes 'text' et 'label'.")

    logging.info(f"🧪 Entraînement d’un modèle contextuel ({model_type.upper()})...")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    preproc = TextPreprocessor()
    texts_cleaned = [preproc.preprocess(t) for t in texts]

    if model_type == "ml":
        vectorizer = TextVectorizer(method="tfidf")
        X = vectorizer.fit_transform_tfidf(texts_cleaned)
        clf = MLTextClassifier(method="logreg")
        clf.train(X, labels)
        return clf, vectorizer

    elif model_type == "dl":
        vectorizer = TextVectorizer(method="word2vec")
        tokenized = [t.split() for t in texts_cleaned]
        vectorizer.train_word2vec(tokenized)
        clf = DLTextClassifier()
        clf.train(tokenized, labels, vectorizer)
        return clf, vectorizer

    else:
        raise ValueError("Méthode non supportée. Utilisez 'ml' ou 'dl'.")

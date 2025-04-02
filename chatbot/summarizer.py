import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def summarize(text: str, max_sentences: int = 5) -> str:
    """
    Résumé extractif basé sur TF-IDF sans modèles pré-entraînés.

    :param text: Texte à résumer
    :param max_sentences: Nombre maximal de phrases à conserver
    :return: Résumé
    """
    logging.info("📝 Résumé extractif maison...")

    # Découpage naïf en phrases
    sentences = [s.strip() for s in text.split('. ') if len(s.strip()) > 10]
    if len(sentences) <= max_sentences:
        return text  # Rien à résumer

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Sélection des meilleures phrases
    top_indices = np.argsort(sentence_scores)[-max_sentences:]
    top_indices.sort()

    best_sentences = [sentences[i] for i in top_indices]
    return '. '.join(best_sentences).strip() + '.'

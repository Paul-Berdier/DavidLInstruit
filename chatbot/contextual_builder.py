import wikipedia
import logging
import spacy
import pandas as pd
from collections import defaultdict

wikipedia.set_lang("fr")
nlp = spacy.load("fr_core_news_md")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_keywords(text: str, top_n: int = 5):
    """
    Utilise spaCy pour extraire les mots-clés pertinents d’un texte utilisateur.
    """
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
    return list(dict.fromkeys(keywords))[:top_n]  # Unique & top N

def build_contextual_corpus(user_input: str, max_per_keyword: int = 2):
    """
    Crée un corpus contextuel basé sur les mots-clés extraits et les résumés Wikipedia associés.
    """
    keywords = extract_keywords(user_input)
    logging.info(f"Mots-clés extraits : {keywords}")

    corpus = []
    labels = []

    for keyword in keywords:
        try:
            search_results = wikipedia.search(keyword)[:max_per_keyword]
            for result in search_results:
                try:
                    logging.info(f"📚 Résumé de : {result}")
                    summary = wikipedia.summary(result, sentences=5)
                    corpus.append(summary)
                    labels.append(keyword)
                except wikipedia.exceptions.DisambiguationError as e:
                    choix = e.options[0]
                    logging.warning(f"⚠️ Ambiguïté sur '{result}', fallback sur '{choix}'")
                    summary = wikipedia.summary(choix, sentences=5)
                    corpus.append(summary)
                    labels.append(keyword)
                except Exception as e:
                    logging.error(f"❌ Échec pour '{result}': {e}")
        except Exception as e:
            logging.error(f"🔍 Erreur lors de la recherche pour '{keyword}': {e}")

    df = pd.DataFrame({"text": corpus, "label": labels})
    return df

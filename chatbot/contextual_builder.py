import wikipedia
import logging
import spacy
import pandas as pd

wikipedia.set_lang("fr")
nlp = spacy.load("fr_core_news_md")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_keywords(text: str, top_n: int = 5):
    """
    Utilise spaCy pour extraire les mots-clés pertinents d’un texte utilisateur.
    """
    doc = nlp(text.lower())
    keywords = [
        token.lemma_ for token in doc
        if token.pos_ in {"NOUN", "PROPN"}
        and not token.is_stop
        and len(token.lemma_) > 2
    ]
    return list(dict.fromkeys(keywords))[:top_n]  # Unique & limité à top_n

def build_contextual_corpus(user_input: str, save: bool = False, max_per_keyword: int = 2) -> pd.DataFrame:
    """
    Crée un DataFrame contextuel basé sur les mots-clés extraits et les résumés Wikipedia associés.
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
                    try:
                        summary = wikipedia.summary(choix, sentences=5)
                        corpus.append(summary)
                        labels.append(keyword)
                    except Exception as err:
                        logging.error(f"❌ Fallback échoué sur '{choix}' : {err}")
                except Exception as e:
                    logging.error(f"❌ Échec pour '{result}': {e}")
        except Exception as e:
            logging.error(f"🔍 Erreur lors de la recherche pour '{keyword}': {e}")

    df = pd.DataFrame({"text": corpus, "label": labels})
    if save:
        df.to_csv("data/context_dataset.csv", index=False)
        logging.info("💾 Dataset contextuel sauvegardé : data/context_dataset.csv")

    return df

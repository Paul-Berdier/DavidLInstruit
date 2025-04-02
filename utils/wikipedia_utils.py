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
    keywords = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
    return list(dict.fromkeys(keywords))[:top_n]

def build_context_dataset(prompt: str, max_results_per_keyword: int = 3, save: bool = True):
    """
    À partir d’un prompt, construit dynamiquement un dataset contextuel avec Wikipédia.
    """
    keywords = extract_keywords(prompt)
    logging.info(f"🔑 Mots-clés : {keywords}")

    texts = []
    labels = []

    for kw in keywords:
        try:
            search_results = wikipedia.search(kw, results=max_results_per_keyword)
            for result in search_results:
                try:
                    summary = wikipedia.summary(result, sentences=5)
                    texts.append(summary)
                    labels.append(kw)
                    logging.info(f"📚 Résumé de : {result}")
                except wikipedia.exceptions.DisambiguationError as e:
                    choix = e.options[0]
                    logging.warning(f"⚠️ Ambiguïté : '{result}' → '{choix}'")
                    try:
                        summary = wikipedia.summary(choix, sentences=5)
                        texts.append(summary)
                        labels.append(kw)
                    except:
                        continue
                except Exception as e:
                    logging.error(f"❌ Erreur avec '{result}': {e}")
        except Exception as e:
            logging.error(f"🔍 Erreur recherche '{kw}': {e}")

    df = pd.DataFrame({"text": texts, "label": labels})
    if save:
        df.to_csv("data/context_dataset.csv", index=False)
        logging.info("✅ Dataset contextuel sauvegardé : data/context_dataset.csv")

    return df

import wikipedia
import pandas as pd
import logging
from utils.preprocessing import TextPreprocessor
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📚 Config de Wikipedia
wikipedia.set_lang("fr")
nlp = spacy.load("fr_core_news_md")
preproc = TextPreprocessor()

def extract_keywords(text, top_n=10):
    doc = nlp(preproc.preprocess(text))
    return [chunk.text.lower() for chunk in doc.noun_chunks][:top_n]

def search_related_topics(keywords, max_pages=30):
    results = set()
    for kw in keywords:
        try:
            found = wikipedia.search(kw)
            results.update(found)
            if len(results) >= max_pages:
                break
        except Exception as e:
            logging.warning(f"Erreur sur recherche '{kw}' : {e}")
    return list(results)[:max_pages]

def fetch_wikipedia_summary(title):
    try:
        summary = wikipedia.summary(title)
        return summary
    except Exception as e:
        logging.warning(f"❌ {title} : {e}")
        return None

def generate_context_dataset(prompt, output_path="data/context_dataset.csv"):
    logging.info(f"🧠 Extraction des mots-clés depuis le prompt : {prompt}")
    keywords = extract_keywords(prompt)
    logging.info(f"🔑 Mots-clés : {keywords}")

    logging.info("🔍 Recherche de sujets liés sur Wikipédia...")
    topics = search_related_topics(keywords)
    logging.info(f"📄 Sujets trouvés : {topics}")

    data = []
    for topic in topics:
        content = fetch_wikipedia_summary(topic)
        if content:
            data.append({"text": content, "label": topic.lower()})

    df = pd.DataFrame(data)
    if df.empty:
        logging.warning("⚠️ Aucun contenu valide récupéré.")
        return None

    df.to_csv(output_path, index=False)
    logging.info(f"✅ Dataset contextuel sauvegardé : {output_path}")
    return df

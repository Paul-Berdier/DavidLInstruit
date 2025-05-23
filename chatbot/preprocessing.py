# chatbot/preprocessing.py

import re
import string
import spacy

# Charge modèle FR
nlp = spacy.load("fr_core_news_md")

def clean_text(text):
    # Minuscule
    text = text.lower()

    # Supprimer HTML, URL, ponctuations
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Supprimer chiffres
    text = re.sub(r"\d+", "", text)

    # Tokenisation + lemmatisation
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_ != "-PRON-"
    ]

    return " ".join(tokens)

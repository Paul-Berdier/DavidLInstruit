import spacy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextPreprocessor:
    """
    Pipeline de nettoyage de texte pour la langue française.
    Utilise spaCy avec le modèle 'fr_core_news_md'.
    """

    def __init__(self):
        logging.info("Chargement du modèle spaCy français...")
        self.nlp = spacy.load("fr_core_news_md")
        self.stopwords = self.nlp.Defaults.stop_words
        logging.info("Modèle chargé avec succès.")

    def preprocess(self, text: str) -> str:
        """
        Nettoie et lemmatise un texte en français.
        Supprime la ponctuation, les chiffres et les stopwords.
        """
        logging.debug(f"Texte brut reçu : {text}")
        doc = self.nlp(text.lower())

        cleaned_tokens = []
        for token in doc:
            if (
                not token.is_punct
                and not token.is_digit
                and not token.is_space
                and token.lemma_ not in self.stopwords
            ):
                cleaned_tokens.append(token.lemma_)

        cleaned_text = " ".join(cleaned_tokens)
        logging.debug(f"Texte nettoyé : {cleaned_text}")
        return cleaned_text

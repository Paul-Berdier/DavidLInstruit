# chatbot/contextual_builder.py

import os
import pandas as pd
import wikipediaapi
import spacy

nlp = spacy.load("fr_core_news_md")

class WikipediaContextBuilder:
    def __init__(self, prompt, language='fr'):
        self.prompt = prompt
        self.language = language
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="DavidLInstruit/1.0"
        )
        self.keywords = []
        self.pages = {}
        self.corpus = []

    def extract_keywords(self, max_keywords=10):
        """🧠 Extrait les mots-clés nominaux/proper nouns du prompt"""
        doc = nlp(self.prompt)
        self.keywords = list({token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]})
        self.keywords = self.keywords[:max_keywords]
        return self.keywords

    def fetch_wikipedia_pages(self, top_n=30):
        """📚 Cherche les résumés des pages Wikipedia associées aux mots-clés"""
        fetched = []
        for kw in self.keywords:
            page = self.wiki.page(kw)
            if page.exists():
                self.pages[kw] = page.summary
                fetched.append((kw, page.summary))
        return fetched[:top_n]

    def build_corpus(self):
        """🧱 Construit un corpus contextualisé pour entraînement"""
        self.corpus = [{"text": summary, "label": keyword}
                       for keyword, summary in self.pages.items()]
        return self.corpus

    def to_dataframe(self, save_path="data/contextual.csv"):
        """💾 Exporte le corpus en DataFrame et le sauvegarde"""
        df = pd.DataFrame(self.corpus)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
        return df

    def train_model(self, model_type="ml"):
        """
        ⚙️ Entraîne un modèle contextuel ML ou DL à partir du corpus construit.
        - Nécessite d'avoir exécuté build_corpus() avant.
        - model_type = 'ml' (TF-IDF + LogisticRegression) ou 'dl' (Tokenizer + LSTM)
        """
        if not self.corpus:
            raise ValueError("❌ Corpus vide — appelle build_corpus() avant d'entraîner le modèle.")

        from chatbot.contextual_model import ContextualModel

        df = pd.DataFrame(self.corpus)
        model = ContextualModel(df, model_type=model_type)
        model.train()
        return model

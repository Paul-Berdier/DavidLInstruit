import os
import pandas as pd
import wikipedia
import spacy

from chatbot.summarize import Summarizer  # Résumeur existant

# 🔽 Téléchargement auto du modèle spaCy anglais (si pas déjà dispo)
os.system("python -m spacy download en_core_web_md")
nlp = spacy.load("en_core_web_md")


class WikipediaContextBuilder:
    def __init__(self, prompt, language='en'):
        self.prompt = prompt
        self.language = language
        wikipedia.set_lang(language)
        self.keywords = []
        self.pages = []      # Titres des pages Wikipedia
        self.raw_texts = []  # Résumés extraits
        self.corpus = []     # Liste de dicts : {"text": résumé, "label": mot-clé}
        self.summarizer = Summarizer()
        self.summarizer.load_or_train()

    def extract_keywords(self, max_keywords=10):
        """🧠 Extrait les mots-clés nominaux et noms propres du prompt utilisateur."""
        doc = nlp(self.prompt)
        self.keywords = list({token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]})
        self.keywords = self.keywords[:max_keywords]
        return self.keywords

    def fetch_wikipedia_pages(self):
        """📚 Essaye une recherche combinée, puis fallback sur chaque mot-clé."""
        combined_query = " ".join(self.keywords)

        try:
            print(f"🔍 Recherche combinée : {combined_query}")
            page_title = wikipedia.search(combined_query, results=1)[0]
            summary = wikipedia.summary(page_title, auto_suggest=False)
            self.pages = [page_title]
            self.raw_texts = [summary]
            print(f"📄 Page combinée trouvée : {page_title}")
        except Exception as e:
            print(f"⚠️ Aucune page combinée trouvée : {e}")
            print("🔁 Fallback sur les mots-clés séparés...")
            self.pages = []
            self.raw_texts = []

            for kw in self.keywords:
                try:
                    page_title = wikipedia.search(kw, results=1)[0]
                    summary = wikipedia.summary(page_title, auto_suggest=False)
                    self.pages.append(page_title)
                    self.raw_texts.append(summary)
                    print(f"📄 Page trouvée pour '{kw}' : {page_title}")
                except Exception as e:
                    print(f"❌ Erreur pour '{kw}' : {e}")

    def build_corpus(self, summarize_each=True):
        """🧱 Construit un corpus résumé à partir des pages Wikipedia."""
        self.corpus = []
        for page_title, text in zip(self.pages, self.raw_texts):
            short_text = (
                self.summarizer.summarize_dl(text, max_lines=2)
                if summarize_each else text
            )
            self.corpus.append({"text": short_text, "label": page_title})
        return self.corpus

    def to_dataframe(self, save_path="data/contextual.csv"):
        """💾 Exporte le corpus en CSV"""
        df = pd.DataFrame(self.corpus)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
        return df

    def generate_answer(self, user_input=None, model_type="ml", max_lines=3):
        """
        🧠 Génère une réponse :
        - S’il n’y a qu’un seul résumé : le résume directement
        - Sinon : prédit le label avec un modèle léger
        """
        if not self.raw_texts:
            return "❌ Aucun résumé disponible."

        if len(self.raw_texts) == 1:
            return self.summarizer.summarize_ml(self.raw_texts[0], max_lines=max_lines)

        # Sinon : modèle contextuel (utile si plusieurs textes)
        from chatbot.contextual_model import ContextualModel
        df = pd.DataFrame(self.corpus)
        model = ContextualModel(df, model_type=model_type)
        model.train()

        predicted_label = model.predict(user_input or self.prompt)

        idx = self.pages.index(predicted_label) if predicted_label in self.pages else 0
        summary = self.raw_texts[idx]

        return self.summarizer.summarize_ml(summary, max_lines=max_lines)

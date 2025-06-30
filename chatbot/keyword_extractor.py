import os
import pandas as pd
import numpy as np
import joblib
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize
from chatbot.preprocessing import clean_text  # suppose que tu as déjà cette fonction

class KeywordExtractor:
    def __init__(self, csv_path="data/keyword_dataset.csv", model_path="models/keyword_extractor.joblib"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.vectorizer = None
        self.model = None

    def _preprocess_dataset(self):
        """
        Transforme le CSV en un DataFrame de lignes mot/label.
        """
        df = pd.read_csv(self.csv_path)
        dataset = []

        for _, row in df.iterrows():
            keywords = row["keywords"]
            if not isinstance(keywords, str) or not keywords.strip():
                print(f"[WARN] No keywords in: {row['text'][:120].strip()}")
                continue

            # Split et nettoyage
            keywords = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            text_cleaned = clean_text(str(row["text"]))
            words = word_tokenize(text_cleaned)

            for word in words:
                if word in string.punctuation or word.isdigit():
                    continue
                label = 1 if word.lower() in keywords else 0
                dataset.append((word.lower(), label))

        df_keywords = pd.DataFrame(dataset, columns=["word", "label"])
        print(f"✅ Dataset préparé : {len(df_keywords)} mots dont {df_keywords['label'].sum()} mots-clés")
        print(df_keywords['label'].value_counts())
        return shuffle(df_keywords, random_state=42)

    def train(self):
        """
        Entraîne un modèle TF-IDF + LogisticRegression pour classer les mots comme mot-clé ou non.
        """
        df = self._preprocess_dataset()
        X = df["word"]
        y = df["label"]

        self.vectorizer = TfidfVectorizer()
        X_vec = self.vectorizer.fit_transform(X)

        self.model = LogisticRegression(class_weight="balanced", max_iter=200)
        self.model.fit(X_vec, y)

        # Rapport
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, digits=3))

        # Sauvegarde
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump((self.vectorizer, self.model), self.model_path)
        print(f"✅ Modèle sauvegardé dans {self.model_path}")

    def load(self):
        """
        Charge le modèle depuis le disque.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("❌ Modèle introuvable, exécute `.train()` d'abord.")
        self.vectorizer, self.model = joblib.load(self.model_path)

    def extract(self, text, top_k=10):
        """
        Prédit les mots-clés à partir d’un texte.
        """
        if self.model is None or self.vectorizer is None:
            self.load()

        text_cleaned = clean_text(text)
        words = list(set(word_tokenize(text_cleaned)))
        words = [w for w in words if w not in string.punctuation and not w.isdigit()]
        X = self.vectorizer.transform(words)
        probs = self.model.predict_proba(X)[:, 1]

        ranked = sorted(zip(words, probs), key=lambda x: x[1], reverse=True)
        keywords = [word for word, prob in ranked[:top_k]]
        return keywords

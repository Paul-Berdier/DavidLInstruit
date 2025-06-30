import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import joblib
import numpy as np
import pandas as pd
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize.punkt import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()
sent_tokenize = tokenizer.tokenize

from chatbot.preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

class Summarizer:
    def __init__(self, data_path="data/summarization_cleaned.csv"):
        self.data_path = data_path
        self.tfidf_path = os.path.join(BASE_DIR, "..", "models", "summarizer", "summary_tfidf_vectorizer.joblib")
        self.ml_model_path = os.path.join(BASE_DIR, "..", "models", "summarizer", "ml_summary_model.joblib")
        self.dl_model_path = "models/summarizer/dl_summary_model.h5"
        self.tokenizer_path = "models/summarizer/tokenizer_summary.joblib"
        self._loaded = False

    def load_or_train(self):
        if self._loaded:
            print("✅ Modèles de résumé déjà chargés.")
            return

        if os.path.exists(self.tfidf_path) and os.path.exists(self.ml_model_path) and \
           os.path.exists(self.dl_model_path) and os.path.exists(self.tokenizer_path):
            print("📦 Chargement des modèles ML & DL de résumé...")
            self.tfidf = joblib.load(self.tfidf_path)
            self.clf_ml = joblib.load(self.ml_model_path)
            self.model_dl = load_model(self.dl_model_path)
            self.tokenizer = joblib.load(self.tokenizer_path)
            self.maxlen = self.model_dl.input_shape[1]
            self._loaded = True
            return

        print("📄 Chargement des données de résumé...")
        # Chargement brut
        df = pd.read_csv(self.data_path).dropna().sample(frac=1.0, random_state=42)

        # ➕ Affichage de la distribution initiale
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.countplot(x=df["label"])
        plt.title("Distribution des labels (initiale)")
        plt.savefig("docs/label_distribution_initiale.png")
        plt.close()

        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        cleaned = [clean_text(t) for t in texts]

        # ➕ Rééquilibrage du dataset
        df_minority = df[df["label"] == 1]
        df_majority = df[df["label"] == 0].sample(n=len(df_minority) * 3, random_state=42)
        df = pd.concat([df_minority, df_majority]).sample(frac=1.0, random_state=42)

        # ➕ Affichage après équilibrage
        sns.countplot(x=df["label"])
        plt.title("Distribution des labels (équilibrée)")
        plt.savefig("docs/label_distribution_equilibree.png")
        plt.close()

        # === ML : TF-IDF + Logistic Regression ===
        print("🧠 Entraînement du modèle ML (TF-IDF + LogisticRegression)...")
        self.tfidf = TfidfVectorizer()
        X_tfidf = self.tfidf.fit_transform(cleaned)
        self.clf_ml = LogisticRegression(class_weight="balanced")
        self.clf_ml.fit(X_tfidf, labels)

        # 🔍 Évaluation sur les données d'entraînement
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

        preds_ml = self.clf_ml.predict(X_tfidf)
        print("\n📊 Rapport de classification (ML) :")
        print(classification_report(labels, preds_ml, digits=3))

        # 🔍 Matrice de confusion
        cm = confusion_matrix(labels, preds_ml)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matrice de confusion (ML)")
        plt.savefig("docs/ml_confusion_matrix.png")
        plt.close()

        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.clf_ml, self.ml_model_path)
        print("✅ Modèle ML entraîné et sauvegardé.")

        # === DL : Embedding + LSTM ===
        print("🧠 Entraînement du modèle DL (Embedding + LSTM)...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(cleaned)
        X_seq = self.tokenizer.texts_to_sequences(cleaned)
        self.maxlen = max(len(seq) for seq in X_seq)
        X_pad = pad_sequences(X_seq, maxlen=self.maxlen)
        y_array = np.array(labels)

        self.model_dl = Sequential()
        self.model_dl.add(Embedding(len(self.tokenizer.word_index) + 1, 64, input_length=self.maxlen))
        self.model_dl.add(LSTM(64))
        self.model_dl.add(Dense(1, activation='sigmoid'))
        self.model_dl.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        history = self.model_dl.fit(
            X_pad, y_array,
            epochs=3,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title("Évolution de l'accuracy (DL)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig("docs/dl_accuracy_curve.png")
        plt.close()

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Évolution de la loss (DL)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("docs/dl_loss_curve.png")
        plt.close()

        self.model_dl.save(self.dl_model_path)
        joblib.dump(self.tokenizer, self.tokenizer_path)
        print("✅ Modèle DL entraîné et sauvegardé.")
        self._loaded = True

    def summarize_ml(self, text, max_lines=3):
        self.load_or_train()
        sentences = sent_tokenize(text)
        cleaned = [clean_text(s) for s in sentences]
        X = self.tfidf.transform(cleaned)
        probs = self.clf_ml.predict_proba(X)[:, 1]
        top_idx = np.argsort(probs)[::-1][:max_lines]
        top_sentences = [sentences[i] for i in sorted(top_idx)]
        return " ".join(top_sentences)

    def summarize_dl(self, text, max_lines=3):
        self.load_or_train()
        sentences = sent_tokenize(text)
        cleaned = [clean_text(s) for s in sentences]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=self.maxlen)
        probs = self.model_dl.predict(padded, verbose=0).ravel()
        top_idx = np.argsort(probs)[::-1][:max_lines]
        top_sentences = [sentences[i] for i in sorted(top_idx)]
        return " ".join(top_sentences)

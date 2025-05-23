# chatbot/classify.py

import os
import pandas as pd
import numpy as np
from chatbot.preprocessing import clean_text

# === ML ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, os

# === DL ===
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Classifier:
    def __init__(self, data_path="data/20news_setfit_cleaned.csv", model_dir="models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        self.ml_model_path = os.path.join(model_dir, "classify_ml.pkl")
        self.dl_model_path = os.path.join(model_dir, "classify_dl.keras")
        self.tokenizer_path = os.path.join(model_dir, "classify_tokenizer.pkl")
        self.label_index_path = os.path.join(model_dir, "label_to_index.pkl")

        self.tfidf = None
        self.clf_ml = None
        self.model_dl = None
        self.tokenizer = None
        self.label_to_index = None
        self.index_to_label = None

        self._loaded = False

    def load_or_train(self):
        if self._loaded:
            print("✅ Modèles de classification déjà chargés.")
            return

        # === Si tous les modèles sont présents, chargement direct ===
        if os.path.exists(self.tfidf_path) and os.path.exists(self.ml_model_path) and \
                os.path.exists(self.dl_model_path) and os.path.exists(self.tokenizer_path) and \
                os.path.exists(self.label_index_path):
            print("📦 Chargement des modèles ML & DL...")
            self.tfidf = joblib.load(self.tfidf_path)
            self.clf_ml = joblib.load(self.ml_model_path)
            self.model_dl = load_model(self.dl_model_path)
            self.tokenizer = joblib.load(self.tokenizer_path)
            self.label_to_index = joblib.load(self.label_index_path)
            self.index_to_label = {i: l for l, i in self.label_to_index.items()}
            self._maxlen = self.model_dl.input_shape[1]
            self._loaded = True
            return

        # === Sinon, charger les données et entraîner ===
        print("📄 Chargement des données de classification...")
        df = pd.read_csv(self.data_path)
        texts = df["cleaned"].astype(str).tolist()
        labels = df["label_text"].astype(str).tolist()
        cleaned_texts = [t for t in texts]

        # === ML ===
        print("🧠 Entraînement du modèle ML (TF-IDF + LogisticRegression)...")
        self.tfidf = TfidfVectorizer(max_features=5000)
        X_tfidf = self.tfidf.fit_transform(cleaned_texts)
        self.clf_ml = LogisticRegression(max_iter=1000)
        self.clf_ml.fit(X_tfidf, labels)
        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.clf_ml, self.ml_model_path)
        print("✅ Modèle ML entraîné et sauvegardé.")

        # === DL ===
        print("🧠 Entraînement du modèle DL (Embedding + LSTM)...")
        self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(cleaned_texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        self._maxlen = max(len(seq) for seq in sequences)
        X_pad = pad_sequences(sequences, maxlen=self._maxlen)

        self.label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        y = np.array([self.label_to_index[label] for label in labels])

        self.model_dl = Sequential()
        self.model_dl.add(Embedding(len(self.tokenizer.word_index) + 1, 64, input_length=self._maxlen))
        self.model_dl.add(LSTM(64))
        self.model_dl.add(Dense(len(self.label_to_index), activation="softmax"))
        self.model_dl.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        import matplotlib.pyplot as plt

        history = self.model_dl.fit(
            X_pad, y,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        # 📈 Courbe d'accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Évolution de l'accuracy - Classification DL")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig("models/classify_dl_accuracy.png")
        plt.close()

        # 📉 Courbe de perte (loss)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Évolution de la loss - Classification DL")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("models/classify_dl_loss.png")
        plt.close()

        self.model_dl.save(self.dl_model_path)
        joblib.dump(self.tokenizer, self.tokenizer_path)
        joblib.dump(self.label_to_index, self.label_index_path)
        print("✅ Modèle DL entraîné et sauvegardé.")

        self._loaded = True

    def predict_ml(self, text):
        self.load_or_train()
        vec = self.tfidf.transform([clean_text(text)])
        return self.clf_ml.predict(vec)[0]

    def predict_dl(self, text):
        self.load_or_train()
        seq = self.tokenizer.texts_to_sequences([clean_text(text)])
        pad_seq = pad_sequences(seq, maxlen=self.model_dl.input_shape[1])
        pred = self.model_dl.predict(pad_seq, verbose=0)
        return self.index_to_label[np.argmax(pred)]
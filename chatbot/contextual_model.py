from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from chatbot.preprocessing import clean_text

class ContextualModel:
    def __init__(self, dataframe, model_type="ml"):
        self.df = dataframe
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.maxlen = None
        self.label_set = sorted(self.df["label"].unique())
        self.label_to_index = {label: i for i, label in enumerate(self.label_set)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}

    def train(self):
        X = self.df["text"].astype(str).apply(clean_text).tolist()
        y = np.array([self.label_to_index[label] for label in self.df["label"]])

        if self.model_type == "ml":
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_vec = self.vectorizer.fit_transform(X)
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_vec, y)

        elif self.model_type == "dl":
            self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(X)
            sequences = self.tokenizer.texts_to_sequences(X)
            self.maxlen = max(len(s) for s in sequences)
            X_pad = pad_sequences(sequences, maxlen=self.maxlen)

            self.model = Sequential()
            self.model.add(Embedding(len(self.tokenizer.word_index) + 1, 64, input_length=self.maxlen))
            self.model.add(LSTM(64))
            self.model.add(Dense(len(self.label_set), activation="softmax"))
            self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            self.model.fit(X_pad, y, epochs=10, verbose=0)

    def predict(self, text):
        cleaned = clean_text(text)

        if self.model_type == "ml":
            vec = self.vectorizer.transform([cleaned])
            pred = self.model.predict(vec)[0]
            return self.index_to_label[pred]

        elif self.model_type == "dl":
            seq = self.tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=self.maxlen)
            pred = self.model.predict(pad, verbose=0)
            return self.index_to_label[np.argmax(pred)]

        return "❌ Aucun modèle chargé"

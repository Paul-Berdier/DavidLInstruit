import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DLTextClassifier:
    def __init__(self, max_words=5000, max_len=200):
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_words = max_words
        self.max_len = max_len

    def preprocess(self, texts, labels):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)

        encoded_labels = self.label_encoder.fit_transform(labels)
        categorical_labels = to_categorical(encoded_labels)

        return padded, categorical_labels

    def build_model(self, output_dim):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(output_dim, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model = model

    def train(self, texts, labels, epochs=10, batch_size=16):
        logging.info("🧠 Prétraitement des données pour le modèle DL...")
        X, y = self.preprocess(texts, labels)
        self.build_model(output_dim=y.shape[1])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        logging.info("🚀 Entraînement du modèle LSTM...")
        callback = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size, callbacks=[callback])

    def predict(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        preds = self.model.predict(padded)
        labels = self.label_encoder.inverse_transform(np.argmax(preds, axis=1))
        return labels

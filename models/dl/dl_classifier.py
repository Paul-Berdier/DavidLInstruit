import logging
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DLTextClassifier:
    def __init__(self, vectorizer=None, maxlen=100):
        self.vectorizer = vectorizer  # Word2Vec (déjà entraîné)
        self.maxlen = maxlen
        self.label_encoder = None
        self.model = None
        self.X_train = None
        self.y_train = None

    def vectorize_sequences(self, sequences, maxlen):
        # Moyenne des vecteurs Word2Vec pour chaque mot dans la séquence
        X = []
        for seq in sequences:
            vectors = [self.vectorizer.get_word_vector(word) for word in seq if self.vectorizer.get_word_vector(word) is not None]
            if vectors:
                X.append(np.mean(vectors, axis=0))
            else:
                X.append(np.zeros(self.vectorizer.model.vector_size))
        return np.array(X)

    def train(self, tokenized_texts, labels, vectorizer):
        self.vectorizer = vectorizer
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        y_cat = to_categorical(y)

        X = self.vectorize_sequences(tokenized_texts, self.maxlen)

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dense(64, activation='relu'),
            Dense(len(set(labels)), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        logging.info("🧠 Entraînement du modèle Deep Learning...")
        self.model.fit(X, y_cat, epochs=10, batch_size=8, verbose=0)

        # Stocke les données utiles pour la génération
        self.X_train = tokenized_texts
        self.y_train = labels

    def predict(self, tokenized_texts):
        X = self.vectorize_sequences(tokenized_texts, self.maxlen)
        preds = self.model.predict(X)
        y_indices = preds.argmax(axis=1)
        return self.label_encoder.inverse_transform(y_indices)

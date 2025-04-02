# main.py
import logging
from utils.wikipedia_utils import build_contextual_dataset
from utils.vectorizer import TextVectorizer
from utils.preprocessing import TextPreprocessor
from models.ml.ml_classifier import MLTextClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 👉 Prompt utilisateur
prompt = input("🎤 Pose ta question à David l’instruit : ")

# 🧠 Générer un dataset contextuel depuis Wikipedia
df = build_contextual_dataset(prompt, save_path=None)
texts = df["text"].tolist()
labels = df["label"].tolist()

# 🧹 Prétraitement
preproc = TextPreprocessor()
texts_cleaned = [preproc.preprocess(t) for t in texts]

# ✨ Vectorisation
vectorizer = TextVectorizer(method="tfidf")
X = vectorizer.fit_transform_tfidf(texts_cleaned)

# 🤖 Entraînement ML
clf = MLTextClassifier(method="logreg")
clf.train(X, labels)

# 🔮 Prédiction immédiate (sur le même prompt pour la démo)
X_prompt = vectorizer.transform_tfidf([preproc.preprocess(prompt)])
prediction = clf.predict(X_prompt)
print(f"\nle savais-tu ? ☝️🤓 {prediction[0]}")

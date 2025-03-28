import logging
from models.ml.ml_classifier import MLTextClassifier
from utils.vectorizer import Vectorizer
from utils.preprocessing import TextPreprocessor
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📥 Charger le dataset contextuel
DATASET_PATH = "data/context_dataset.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset contextuel non trouvé à {DATASET_PATH}")

logging.info("📥 Chargement du dataset...")
df = pd.read_csv(DATASET_PATH)
texts, labels = df["text"].tolist(), df["label"].tolist()

# Nettoyage de classe : au moins 2 exemples
class_counts = Counter(labels)
texts_filtered = [t for t, l in zip(texts, labels) if class_counts[l] >= 2]
labels_filtered = [l for l in labels if class_counts[l] >= 2]

# Preprocessing + Vectorisation
preproc = TextPreprocessor()
vectorizer = Vectorizer(method="tfidf")
texts_cleaned = [preproc.preprocess(t) for t in texts_filtered]
X = vectorizer.fit_transform_tfidf(texts_cleaned)

# Entraînement ML
logging.info("⚙️ Entraînement du modèle ML (Logistic Regression)")
clf_ml = MLTextClassifier(method="logreg")
clf_ml.train(X, labels_filtered)
clf_ml.save_model("models/ml/model.joblib")
vectorizer.save("models/ml/vectorizer.joblib")

logging.info("✅ Modèle ML entraîné et sauvegardé avec succès.")

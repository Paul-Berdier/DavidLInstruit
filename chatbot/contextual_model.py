from chatbot.contextual_builder import build_contextual_corpus
from utils.preprocessing import TextPreprocessor
from utils.vectorizer import TextVectorizer
from models.ml.ml_classifier import MLTextClassifier

from sklearn.model_selection import train_test_split
from collections import Counter

def predict_contextual_label(user_input: str):
    """
    Entraîne un modèle temporaire sur un corpus Wikipedia contextuel,
    puis prédit la catégorie (mot-clé) du texte utilisateur.
    """
    # Étape 1 : Corpus contextuel
    df = build_contextual_corpus(user_input)
    if df.empty or df["label"].nunique() < 2:
        return "❌ Corpus insuffisant pour générer une réponse contextuelle."

    # Étape 2 : Prétraitement
    preproc = TextPreprocessor()
    cleaned_corpus = [preproc.preprocess(text) for text in df["text"]]

    # Étape 3 : Vectorisation
    vectorizer = TextVectorizer()
    X = vectorizer.fit_transform_tfidf(cleaned_corpus)

    # Étape 4 : Entraînement
    X_train, _, y_train, _ = train_test_split(X, df["label"], test_size=0.2)
    class_counts = Counter(y_train)
    if any(count < 2 for count in class_counts.values()):
        return f"❌ Pas assez de données pour entraîner un modèle robuste : {class_counts}"

    clf = MLTextClassifier("logreg")
    clf.train(X_train, y_train)

    # Étape 5 : Prédiction du texte utilisateur
    user_clean = preproc.preprocess(user_input)
    user_vec = vectorizer.transform_tfidf([user_clean])
    prediction = clf.predict(user_vec)[0]

    return f"🧠 Selon moi, tu parles de **{prediction}**."

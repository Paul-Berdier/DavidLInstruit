import logging
import random
from utils.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INTRO_PHRASES = [
    "tu le savais-tu ? ☝️🤓",
    "oui je le savais-tu ☝️🤓",
    "mais enfin...tu le savais-tu ? ☝️🤓",
    "tiens-toi bien : je le savais-tu... ☝️🤓"
]


def respond_to(prompt: str, model, vectorizer) -> str:
    """
    Prend un prompt utilisateur et renvoie une réponse prédite à partir d’un modèle et d’un vectorizer fournis.

    :param prompt: Le texte de l'utilisateur.
    :param model: Le modèle ML ou DL entraîné.
    :param vectorizer: Le vectoriseur utilisé lors de l'entraînement.
    :return: Une phrase contextuelle.
    """
    logging.info("📨 Nettoyage du prompt utilisateur...")
    preproc = TextPreprocessor()
    prompt_clean = preproc.preprocess(prompt)

    if vectorizer.method == "tfidf":
        X_prompt = vectorizer.transform_tfidf([prompt_clean])
    elif vectorizer.method == "word2vec":
        X_prompt = [prompt_clean.split()]
    else:
        raise ValueError("Méthode de vectorisation inconnue.")

    prediction = model.predict(X_prompt)[0]

    # Contexte (hack temporaire)
    if hasattr(model, "X_train") and hasattr(model, "y_train"):
        textes = model.X_train
        labels = model.y_train
        contexte = [t for t, l in zip(textes, labels) if l == prediction]
        if contexte:
            sample = random.choice(contexte)
            return f"{random.choice(INTRO_PHRASES)} {sample}"

    return f"{random.choice(INTRO_PHRASES)} Heu… je n’ai rien trouvé de cohérent à te dire."

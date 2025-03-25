import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump, load
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MLTextClassifier:
    """
    Classe pour entraîner et utiliser des modèles de classification textuelle.
    Supporte Logistic Regression et Random Forest.
    """

    def __init__(self, model_type="logreg"):
        if model_type == "logreg":
            self.model = LogisticRegression(max_iter=1000)
            self.params = {'C': [0.1, 1, 10]}
        elif model_type == "rf":
            self.model = RandomForestClassifier()
            self.params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        else:
            raise ValueError("model_type must be 'logreg' or 'rf'")

        self.grid = None
        logging.info(f"Initialisation du modèle : {model_type}")

    def train(self, X, y):
        """
        Entraîne le modèle avec GridSearchCV, en adaptant dynamiquement le nombre de folds
        en fonction du minimum de samples par classe.
        """
        label_counts = Counter(y)
        min_class_count = min(label_counts.values())
        cv = min(5, min_class_count)

        if cv < 2:
            raise ValueError(
                f"Pas assez d’échantillons par classe pour faire une validation croisée stratifiée. "
                f"Minimum requis : 2 par classe, reçu : {label_counts}"
            )

        logging.info(f"Entraînement du modèle avec GridSearchCV (cv={cv})...")

        self.grid = GridSearchCV(
            self.model,
            self.params,
            cv=cv,
            scoring='accuracy',
            verbose=1
        )
        self.grid.fit(X, y)
        logging.info(f"Meilleurs paramètres : {self.grid.best_params_}")

    def evaluate(self, X_test, y_test):
        """
        Affiche un rapport de classification.
        """
        if self.grid is None:
            raise ValueError("Le modèle n’a pas encore été entraîné.")
        preds = self.grid.predict(X_test)
        print(classification_report(y_test, preds))

    def predict(self, X):
        """
        Prédit une ou plusieurs classes.
        """
        if self.grid is None:
            raise ValueError("Le modèle n’a pas encore été entraîné.")
        return self.grid.predict(X)

    def save_model(self, path: str):
        """
        Sauvegarde le modèle sur disque.
        """
        if self.grid is None:
            raise ValueError("Aucun modèle à sauvegarder.")
        dump(self.grid, path)
        logging.info(f"Modèle sauvegardé à : {path}")

    def load_model(self, path: str):
        """
        Charge un modèle depuis le disque.
        """
        self.grid = load(path)
        logging.info(f"Modèle chargé depuis : {path}")

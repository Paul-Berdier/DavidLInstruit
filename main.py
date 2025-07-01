import os
import time
import nltk
nltk.download("punkt")

# ANSI couleurs
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def title(text):
    print(f"\n{YELLOW}{'='*60}\n{text.upper()}\n{'='*60}{RESET}")

def menu():
    print(f"\n{CYAN}📦 CHOISISSEZ UNE OU PLUSIEURS ÉTAPES (séparées par une virgule) :{RESET}")
    print("1. 📂 PRÉPARER LES DONNÉES (téléchargement + nettoyage CSV)")
    print("2. 🧠 ENTRAÎNER LES MODÈLES DE CLASSIFICATION")
    print("3. 📝 ENTRAÎNER LES MODÈLES DE RÉSUMÉ")
    print("4. 🔑 ENTRAÎNER LE MODÈLE DE MOTS-CLÉS")
    print("5. 🌍 CHARGER ET TESTER LA TRADUCTION (Argos Translate)")
    print("6. 🚀 LANCER L'APPLICATION (interface FastAPI en localhost)")
    print("0. ❌ QUITTER")
    return input("\n👉 Choix (ex: 1,3) : ")

def run_prepare_data():
    title("PRÉPARATION DES DONNÉES : TÉLÉCHARGEMENT & NETTOYAGE")
    scripts = [
        "data/import_labeling_csv.py",
        "data/import_parquet_csv.py",
        "data/import_summarize_csv.py",
        "data/cleanned_csv.py"
    ]
    for script in scripts:
        print(f"📂 Exécution de {script}...")
        exit_code = os.system(f"python {script}")
        if exit_code != 0:
            print(f"{RED}❌ Erreur lors de l'exécution de {script}.{RESET}")
            return
    print(f"{GREEN}✅ Tous les scripts de préparation ont été exécutés avec succès.{RESET}")

def run_classify():
    run_prepare_data()
    title("ENTRAÎNEMENT DU CLASSIFIEUR")
    from chatbot.classify import Classifier
    classifier = Classifier()
    classifier.load_or_train()

def run_summarize():
    run_prepare_data()
    title("ENTRAÎNEMENT DU RÉSUMEUR")
    from chatbot.summarize import Summarizer
    summarizer = Summarizer()
    summarizer.load_or_train()

def run_keyword():
    run_prepare_data()
    title("ENTRAÎNEMENT DU MODÈLE DE MOTS-CLÉS")
    from chatbot.keyword_extractor import KeywordExtractor
    extractor = KeywordExtractor()
    extractor.train()

def run_app():
    title("LANCEMENT DE L'INTERFACE FASTAPI")
    time.sleep(1)
    os.system("uvicorn chatbot.interface.app:app --reload")

def run_translate():
    import urllib.request
    import argostranslate.package
    from termcolor import colored

    title("CHARGEMENT ET TEST DES MODÈLES DE TRADUCTION")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    urls = {
        "fr_en": "https://data.argosopentech.com/argospm/v1/translate-fr_en-1_9.argosmodel",
        "en_fr": "https://data.argosopentech.com/argospm/v1/translate-en_fr-1_9.argosmodel"
    }

    paths = {key: os.path.join(model_dir, os.path.basename(url)) for key, url in urls.items()}

    for key, url in urls.items():
        path = paths[key]
        if not os.path.exists(path):
            print(f"⬇️ Téléchargement du modèle {key}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"✅ Modèle {key} téléchargé : {path}")
            except Exception as e:
                print(f"❌ Erreur lors du téléchargement de {url} : {e}")
                return

    print("📦 Installation du modèle FR → EN...")
    argostranslate.package.install_from_path(paths["fr_en"])
    print("📦 Installation du modèle EN → FR...")
    argostranslate.package.install_from_path(paths["en_fr"])

    print("✅ Modèles installés avec succès.")

    try:
        from chatbot.translation import translate_fr_to_en, translate_en_to_fr
        print(colored("Traduction FR → EN :", "green"), translate_fr_to_en("Bonjour, comment vas-tu ?"))
        print(colored("Traduction EN → FR :", "green"), translate_en_to_fr("What are the benefits of learning AI?"))
    except Exception as e:
        print(colored("❌ Erreur : " + str(e), "red"))

if __name__ == "__main__":
    while True:
        choix = menu().replace(" ", "").split(",")
        if "0" in choix:
            print(f"{RED}👋 À bientôt !{RESET}")
            break
        if "1" in choix:
            run_prepare_data()
        if "2" in choix:
            run_classify()
        if "3" in choix:
            run_summarize()
        if "4" in choix:
            run_keyword()
        if "5" in choix:
            run_translate()
        if "6" in choix:
            run_app()

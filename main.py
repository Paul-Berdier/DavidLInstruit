import os
import time

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
    print("1. 🧠 ENTRAÎNER LES MODÈLES DE CLASSIFICATION (classify)")
    print("2. 📝 ENTRAÎNER LES MODÈLES DE RÉSUMÉ (summarize)")
    print("3. 🌍 CHARGER ET TESTER LA TRADUCTION (Argos Translate)")
    print("4. 🚀 LANCER L'APPLICATION (interface FastAPI en localhost)")
    print("0. ❌ QUITTER")
    return input("\n👉 Choix (ex: 1,3) : ")

def run_classify():
    title("ENTRAÎNEMENT DU CLASSIFIEUR")
    from chatbot.classify import Classifier
    classifier = Classifier()
    classifier.load_or_train()

def run_summarize():
    title("ENTRAÎNEMENT DU RÉSUMEUR")
    from chatbot.summarize import Summarizer
    summarizer = Summarizer()
    summarizer.load_or_train()

def run_app():
    title("LANCEMENT DE L'INTERFACE FASTAPI")
    time.sleep(1)
    os.system("uvicorn chatbot.interface.app:app --reload")

def run_translate():
    import argostranslate.package

    title("CHARGEMENT ET TEST DES MODÈLES DE TRADUCTION")

    model_path_fr_en = "models/translate-fr_en-1_9.argosmodel"
    model_path_en_fr = "models/translate-en_fr-1_9.argosmodel"

    if not os.path.exists(model_path_en_fr):
        raise FileNotFoundError(f"❌ Le fichier {model_path_en_fr} est introuvable.")
    if not os.path.exists(model_path_fr_en):
        raise FileNotFoundError(f"❌ Le fichier {model_path_fr_en} est introuvable.")

    print("📦 Installation du modèle FR ↔ EN...")
    argostranslate.package.install_from_path(model_path_fr_en)
    print("📦 Installation du modèle EN ↔ FR...")
    argostranslate.package.install_from_path(model_path_en_fr)

    print("✅ Modèle installé avec succès.")

    try:
        from chatbot.translation import translate_fr_to_en, translate_en_to_fr
        print(GREEN + "Traduction FR → EN :", translate_fr_to_en("Bonjour, comment vas-tu ?") + RESET)
        print(GREEN + "Traduction EN → FR :", translate_en_to_fr("What are the benefits of learning AI?") + RESET)
    except Exception as e:
        print(RED + "❌ Erreur : " + str(e) + RESET)

if __name__ == "__main__":
    while True:
        choix = menu().replace(" ", "").split(",")
        if "0" in choix:
            print(f"{RED}👋 À bientôt !{RESET}")
            break
        if "1" in choix:
            run_classify()
        if "2" in choix:
            run_summarize()
        if "3" in choix:
            run_translate()
        if "4" in choix:
           run_app()

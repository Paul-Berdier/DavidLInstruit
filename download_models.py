import os
import shutil
from huggingface_hub import snapshot_download
import argostranslate.package

def download_and_copy_models():
    repo_id = "OnlySmaiil/david-models"
    download_dir = "tmp_models"
    source_models_dir = os.path.join(download_dir, "models")
    target_models_dir = "models"

    print("📥 Téléchargement des fichiers depuis Hugging Face...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=download_dir,
        repo_type="model",
        local_dir_use_symlinks=False
    )

    # Supprimer les anciens modèles s'ils existent
    if os.path.exists(target_models_dir):
        shutil.rmtree(target_models_dir)

    # Copier les nouveaux
    shutil.copytree(source_models_dir, target_models_dir)
    print("✅ Modèles copiés dans ./models")

    # Nettoyage
    shutil.rmtree(download_dir)
    print("🧹 Nettoyage terminé.")

def install_argos_models():
    print("🔧 Installation des modèles de traduction Argos Translate...")
    for file in os.listdir("models"):
        if file.endswith(".argosmodel"):
            try:
                path = os.path.join("models", file)
                argostranslate.package.install_from_path(path)
                print(f"✅ Modèle installé : {file}")
            except Exception as e:
                print(f"❌ Erreur pour {file} : {e}")

if __name__ == "__main__":
    download_and_copy_models()
    install_argos_models()
    print("🎉 Tous les modèles sont prêts.")

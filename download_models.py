import os
import shutil
from huggingface_hub import snapshot_download

# 🔁 Télécharger tout le repo contenant le dossier "models/"
repo_id = "OnlySmaiil/david-models"  # <-- remplace par ton identifiant complet
download_dir = "tmp_models"

print("📥 Téléchargement des fichiers depuis Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    local_dir=download_dir,
    repo_type="model",
    local_dir_use_symlinks=False
)

# 📂 Copier le dossier "models" du repo téléchargé à la racine du projet
source_models_dir = os.path.join(download_dir, "models")
target_models_dir = "models"

# Supprimer les anciens modèles s'ils existent
if os.path.exists(target_models_dir):
    shutil.rmtree(target_models_dir)

shutil.copytree(source_models_dir, target_models_dir)
print("✅ Modèles copiés dans ./models")

# Optionnel : nettoyage du dossier temporaire
shutil.rmtree(download_dir)
print("🧹 Nettoyage terminé.")

from datasets import load_dataset
import pandas as pd

# 📥 Chargement du dataset
dataset = load_dataset("SetFit/20_newsgroups")

# 📄 Conversion en DataFrame
df = pd.DataFrame(dataset['train'])

# 🧾 Associer manuellement les IDs aux labels textuels
# On récupère toutes les combinaisons distinctes
label_mapping = {
    row['label']: row['label_text'] for row in dataset['train'] if 'label_text' in row
}

# ✅ Ajout de la colonne texte si elle n'existe pas déjà
if 'label_text' not in df.columns:
    df['label_text'] = df['label'].map(label_mapping)

# 💾 Export en CSV
df.to_csv("data/20news_setfit.csv", index=False, encoding="utf-8")
print("✅ Exporté avec succès sous 20news_setfit.csv")

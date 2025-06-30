# parquet_to_csv.py

import pandas as pd

# 📂 Noms des fichiers à lire
files = [
    "data/validation-00000-of-00001 (1).parquet",
    "data/test-00000-of-00001.parquet",
    "data/train-00000-of-00001.parquet"
]

# ✅ Colonnes utiles à conserver
columns_keep = ['text', 'keywords', 'topic']

# 🧩 Lire et filtrer les fichiers
dfs = []
for file in files:
    print(f"🔄 Lecture de : {file}")
    df = pd.read_parquet(file)[columns_keep]
    dfs.append(df)

# 🧱 Combiner les données
df_combined = pd.concat(dfs, ignore_index=True)
print(f"✅ Total des lignes combinées : {len(df_combined)}")

# 💾 Sauvegarde en CSV
output_file = "data/keyword_dataset.csv"
df_combined.to_csv(output_file, index=False, encoding="utf-8")
print(f"📁 Fichier CSV généré : {output_file}")

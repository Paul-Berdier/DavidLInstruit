# cleaned_csv.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from chatbot.preprocessing import clean_text


def process_and_save(input_path, text_column, output_path, keep_cols=None):
    if not os.path.exists(input_path):
        print(f"❌ Fichier non trouvé : {input_path}")
        return

    print(f"📄 Chargement de {input_path}...")
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        print(f"❌ Colonne '{text_column}' introuvable dans {input_path}")
        return

    df = df.dropna(subset=[text_column])
    df["cleaned"] = df[text_column].astype(str).apply(clean_text)

    if keep_cols:
        keep = [col for col in keep_cols if col in df.columns] + ["cleaned"]
        df = df[keep]

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Fichier nettoyé sauvegardé : {output_path}")
    print(f"📊 Colonnes sauvegardées : {list(df.columns)}\n")


if __name__ == "__main__":
    # Résumé (texte long + summary)
    process_and_save(
        input_path="data/summarization_texts.csv",
        text_column="text",
        output_path="data/summarization_cleaned.csv",
        keep_cols=["text", "summary"]
    )

    # Classification (texte + label)
    process_and_save(
        input_path="data/20news_setfit.csv",
        text_column="text",
        output_path="data/20news_setfit_cleaned.csv",
        keep_cols=["text", "label"]
    )

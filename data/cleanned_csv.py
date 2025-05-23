# cleaned_csv.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from chatbot.preprocessing import clean_text
import os


def process_and_save(input_path, text_column, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Fichier non trouvé : {input_path}")
        return

    print(f"📄 Chargement de {input_path}...")
    df = pd.read_csv(input_path).dropna(subset=[text_column])
    df["cleaned"] = df[text_column].astype(str).apply(clean_text)

    df.to_csv(output_path, index=False)
    print(f"✅ Fichier nettoyé sauvegardé : {output_path}")


if __name__ == "__main__":
    process_and_save(
        input_path="data/summarization_texts.csv",
        text_column="text",
        output_path="data/summarization_cleaned.csv"
    )

    process_and_save(
        input_path="data/20news_setfit.csv",
        text_column="text",
        output_path="data/20news_setfit_cleaned.csv"
    )

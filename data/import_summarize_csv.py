from datasets import load_dataset
import pandas as pd

print("📥 Chargement de CNN/DailyMail (5000)...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")

data = [{"text": item["article"], "summary": item["highlights"]} for item in dataset]

df = pd.DataFrame(data)
df.to_csv("data/summarization_texts.csv", index=False, encoding="utf-8")

print("✅ CSV généré : data/summarization_texts.csv")

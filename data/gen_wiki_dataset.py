import wikipedia
import pandas as pd
from collections import Counter

wikipedia.set_lang("fr")

# Sujets + labels
sujets = {
    "philosophie": "esprit",
    "psychologie": "esprit",
    "intelligence artificielle": "machine",
    "robotique": "machine",
    "biologie": "nature",
    "écologie": "nature",
    "histoire": "passé",
    "antiquité": "passé",
    "Seconde Guerre mondiale": "passé",
    "climat": "nature"
}

corpus = []
labels = []
stats = Counter()

for sujet, label in sujets.items():
    try:
        print(f"📚 {sujet}")
        try:
            texte = wikipedia.summary(sujet, sentences=5)
        except wikipedia.exceptions.DisambiguationError as e:
            # Fallback : on prend le premier résultat proposé
            choix = e.options[0]
            print(f"⚠️ Ambigu : '{sujet}' → fallback sur : '{choix}'")
            texte = wikipedia.summary(choix, sentences=5)

        corpus.append(texte)
        labels.append(label)
        stats[label] += 1

    except Exception as e:
        print(f"❌ {sujet} : {e}")

# Résumé
print("\n✅ Récap des classes récupérées :")
for lbl, count in stats.items():
    print(f"- {lbl} : {count} textes")

# Check : au moins 2 par classe
valid_labels = [lbl for lbl, count in stats.items() if count >= 2]
if len(valid_labels) < 2:
    print("❌ Pas assez de classes valides. Ajoute plus de sujets.")
else:
    df = pd.DataFrame({"text": corpus, "label": labels})
    df.to_csv("data/wiki_dataset.csv", index=False, encoding="utf-8")
    print("✅ Dataset Wikipédia généré avec succès.")

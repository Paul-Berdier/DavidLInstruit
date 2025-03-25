import gradio as gr
from utils.preprocessing import TextPreprocessor
from utils.vectorizer import TextVectorizer
from models.ml.ml_classifier import MLTextClassifier
from chatbot.response_generator import respond_to

import pandas as pd
import joblib

# Chargement des composants
preproc = TextPreprocessor()
vectorizer = TextVectorizer()
clf = MLTextClassifier("logreg")
clf.load_model("models/ml/model.joblib")  # Tu dois sauvegarder le modèle après train


def analyser(prompt):
    cleaned = preproc.preprocess(prompt)
    vec = vectorizer.transform_tfidf([cleaned])
    prediction = clf.predict(vec)[0]
    return f"Classe prédite : **{prediction}**"


def repondre(prompt):
    return respond_to(prompt)


# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("### 👨‍🏫 Bienvenue, je suis **David l'instruit** !")

    with gr.Row():
        prompt_input = gr.Textbox(label="Pose ta question ou entre un texte")

    with gr.Row():
        btn_classify = gr.Button("📊 Classer le texte")
        btn_wiki = gr.Button("📚 Chercher sur Wikipedia")

    output = gr.Markdown()

    btn_classify.click(fn=analyser, inputs=prompt_input, outputs=output)
    btn_wiki.click(fn=repondre, inputs=prompt_input, outputs=output)

demo.launch()

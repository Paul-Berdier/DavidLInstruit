import gradio as gr
import logging
import random
import os

from chatbot.contextual_builder import build_contextual_corpus
from chatbot.contextual_model import train_contextual_model
from chatbot.response import respond_to
from chatbot.summarizer import summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Récupération du chemin absolu de l’image
img_path = os.path.join(os.path.dirname(__file__), "assets", "david_nerd_1.png")

# 🧑‍🏫 Intro de David
INTRO_PHRASES = [
    "tu le savais-tu ? ☝️🤓",
    "oui je le savais-tu ☝️🤓",
    "mais enfin...tu le savais-tu ? ☝️🤓",
    "tiens-toi bien : je le savais-tu... ☝️🤓"
]


def repondre(prompt: str, max_lines: int = 5, mode: str = "ml") -> str:
    logging.info(f"🧠 Extraction des mots-clés depuis le prompt : {prompt}")

    df = build_contextual_corpus(prompt, max_per_keyword=2)

    if df.empty or df["label"].nunique() < 2:
        return f"{random.choice(INTRO_PHRASES)} Je n'ai rien trouvé d'intéressant sur ce sujet. Essaie autre chose !"

    model, vectorizer = train_contextual_model(df, model_type=mode)
    response_raw = respond_to(prompt, model, vectorizer)
    résumé = summarize(response_raw, max_sentences=max_lines)

    intro = random.choice(INTRO_PHRASES)
    return f"{intro} {résumé}"

# 🎨 Interface Gradio
with gr.Blocks(title="David l'instruit") as demo:
    gr.Markdown("# 🧑‍🏫 David l'instruit")
    gr.Image(img_path, label="Le visage énervant de David", show_label=False, width=200)
    gr.Markdown("### Le bot intelligent, légèrement pénible, mais diablement cultivé.\n**Pose-lui une question, il répondra... à sa manière.**")

    with gr.Row():
        prompt = gr.Textbox(label="Ta question", placeholder="Ex: Comment l'IA influence-t-elle la musique ?")
        ligne_max = gr.Slider(minimum=1, maximum=10, value=3, label="Nombre max de phrases")
        mode = gr.Radio(choices=["ml", "dl"], value="ml", label="Méthode de prédiction")

    btn = gr.Button("Demande à David 🧠")
    sortie = gr.Textbox(label="Réponse de David")

    btn.click(fn=repondre, inputs=[prompt, ligne_max, mode], outputs=sortie)

# 🏁 Lancer l'appli
if __name__ == "__main__":
    demo.launch()

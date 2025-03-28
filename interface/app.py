import gradio as gr
from chatbot.response_generator import respond_to
from data.gen_context_dataset import generate_context_dataset
import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def entrainer_model(prompt_utilisateur):
    # Génération du dataset à partir du thème
    df = generate_context_dataset(prompt_utilisateur)
    if df is None or len(df) < 4:
        return "❌ Pas assez de contenu récupéré pour entraîner un modèle."

    # Lancement du script d'entraînement
    logging.info("🔁 Entraînement en cours...")
    result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(result.stderr)
        return "❌ Erreur lors de l'entraînement. Consulte les logs."

    return f"✅ Modèle entraîné avec succès sur {len(df)} textes !"


def repondre(prompt, nb_lignes):
    return respond_to(prompt, max_lines=nb_lignes)


# 🧠 Interface Gradio
with gr.Blocks(title="David l'instruit") as demo:
    gr.Markdown("### 🧑‍🏫 David l’instruit\n\nLe chatbot intelligent et pénible. Oui je le savais-tu ☝️🤓")

    with gr.Row():
        prompt_theme = gr.Textbox(label="Décris un thème pour entraîner David",
                                  placeholder="ex : l'impact des robots dans l'éducation")
        bouton_entrainer = gr.Button("Créer un modèle contextuel 🛠️")

    sortie_entrainement = gr.Textbox(label="🧪 État d'entraînement", interactive=False)
    bouton_entrainer.click(fn=entrainer_model, inputs=[prompt_theme], outputs=[sortie_entrainement])

    gr.Markdown("---")

    with gr.Row():
        prompt_question = gr.Textbox(label="Pose ta question à David", placeholder="ex : Quel est le rôle de l'IA ?")
        max_lignes = gr.Slider(minimum=1, maximum=10, value=3, label="📏 Réponse maximale (lignes)")

    bouton_repondre = gr.Button("Réponds, David ! 🤖")
    sortie = gr.Textbox(label="💬 Réponse de David", lines=5)
    bouton_repondre.click(fn=repondre, inputs=[prompt_question, max_lignes], outputs=[sortie])

# 🔥 Lancer l'app
if __name__ == "__main__":
    demo.launch()

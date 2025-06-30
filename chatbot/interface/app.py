import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# === Fonctions de ton projet ===
from chatbot.classify import Classifier, simplify_label
from chatbot.summarize import Summarizer
from chatbot.contextual_builder import WikipediaContextBuilder
from chatbot.translation import translate_fr_to_en, translate_en_to_fr

app = FastAPI()

# === Configuration static et templates ===
app.mount("/static", StaticFiles(directory="chatbot/interface/static"), name="static")
app.mount("/assets", StaticFiles(directory="chatbot/interface/assets"), name="assets")
templates = Jinja2Templates(directory="chatbot/interface/templates")

# === Modèles à charger une fois ===
classifier = Classifier()
summarizer = Summarizer()

@app.get("/", response_class=HTMLResponse)
async def show_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})

@app.post("/ask", response_class=HTMLResponse)
async def handle_question(
    request: Request,
    question: str = Form(...),
    mode: str = Form(...)
):

    if mode == "classify":
        question_en = translate_fr_to_en(question)
        pred_ml_en = classifier.predict_ml(question_en)
        pred_dl_en = classifier.predict_dl(question_en)

        # simplification des labels
        simplified_ml = simplify_label(pred_ml_en)
        simplified_dl = simplify_label(pred_dl_en)

        response = f"ML : {translate_en_to_fr(simplified_ml)} | DL : {translate_en_to_fr(simplified_dl)}"


    elif mode == "summarize":
        question_en = translate_fr_to_en(question)
        summary_ml_en = summarizer.summarize_ml(question_en)
        summary_dl_en = summarizer.summarize_dl(question_en)
        response = f"ML : {translate_en_to_fr(summary_ml_en)}<br>DL : {translate_en_to_fr(summary_dl_en)}"

    elif mode == "wikipedia":
        builder = WikipediaContextBuilder(question)
        builder.extract_keywords()
        builder.fetch_wikipedia_pages()
        builder.build_corpus()
        model = builder.train_model(model_type="ml")
        result = model.predict(question)
        summary = builder.pages.get(result, "❌ Aucun résumé trouvé.")
        response = f"📚 Sujet identifié : <b>{result}</b><br>{summary}"

    else:
        response = "❌ Mode inconnu"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "response": response,
        "question": question  # ⬅️ ajoute ça
    })

# === Pour lancement local direct ===
if __name__ == "__main__":
    uvicorn.run("interface.app:app", host="127.0.0.1", port=8000, reload=True)

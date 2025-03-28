import wikipedia
import logging

wikipedia.set_lang("fr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def respond_to(prompt: str, max_sentences: int = 3) -> str:
    """
    Cherche sur Wikipédia un résumé limité à max_sentences.
    Reformule et retourne une réponse propre.
    """
    try:
        logging.info(f"Recherche de '{prompt}' sur Wikipedia...")
        try:
            summary = wikipedia.summary(prompt, sentences=max_sentences)
        except wikipedia.exceptions.DisambiguationError as e:
            choix = e.options[0]
            logging.warning(f"Ambigu : fallback sur '{choix}'")
            summary = wikipedia.summary(choix, sentences=max_sentences)

        return summary

    except wikipedia.exceptions.PageError:
        return f"Désolé, je n’ai trouvé aucune information sur **{prompt}**."

    except Exception as e:
        logging.error(f"Erreur inattendue : {e}")
        return "Une erreur est survenue lors de ma recherche. 😓"

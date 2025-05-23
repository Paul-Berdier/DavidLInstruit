# chatbot/translation.py

from argostranslate.translate import get_installed_languages

langs = get_installed_languages()

fr = next((l for l in langs if l.code == "fr"), None)
en = next((l for l in langs if l.code == "en"), None)

if not fr or not en:
    raise Exception("❌ Les langues FR ou EN ne sont pas installées dans Argos Translate.")

translator_fr_to_en = fr.get_translation(en)
translator_en_to_fr = en.get_translation(fr)

def translate_fr_to_en(text):
    """🇫🇷→🇬🇧 Traduction FR → EN"""
    return translator_fr_to_en.translate(text)

def translate_en_to_fr(text):
    """🇬🇧→🇫🇷 Traduction EN → FR"""
    return translator_en_to_fr.translate(text)

FROM python:3.10-slim

# Dépendances système utiles (si tu as spaCy, tesseract, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dossier app
WORKDIR /app

# Copie des fichiers
COPY . .

# Install des dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Téléchargement des modèles Hugging Face (script à toi)
RUN python download_models.py

# Lancer l'app
CMD ["uvicorn", "chatbot.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]

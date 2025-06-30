# 📚 DavidLInstruit

**DavidLInstruit** est un assistant IA modulaire combinant résumé, classification, extraction de mots-clés, génération de réponse via Wikipedia et interface web, le tout propulsé par des modèles Machine Learning (ML) **et Deep Learning (DL)**.

---

## 🚀 Fonctionnalités

| Fonction                   | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| 🔍 Classification NLP      | Classifie un texte dans un thème prédéfini (via ML ou DL) |
| ✂️ Résumé de texte         | Résume automatiquement un texte long (via ML ou DL)       |
| 🧠 Extraction de mots-clés | Repère les termes importants du texte                     |
| 🌐 Recherche Wikipedia     | Cherche un contexte et construit un corpus autour         |
| 🤖 Génération de réponse   | Produit une réponse intelligente à la volée               |
| 🌍 Traduction automatique  | Basée sur Argos Translate (FR/EN)                         |
| 💻 Interface locale        | FastAPI avec HTML/CSS, ou Gradio                          |

---

## 🧠 Modèles utilisés

### 1. 🔍 **Classification de texte**

* **Version ML** :

  * `TfidfVectorizer` + `LogisticRegression`
  * GridSearchCV pour l’optimisation des hyperparamètres
* **Version DL** :

  * Embedding Word2Vec (préentraîné)
  * LSTM (Keras) avec Dense Softmax
  * Entraîné sur le même corpus que le ML

#### 📊 Résultats du modèle ML (Classification)

| Paramètre       | Valeur                               |
| --------------- | ------------------------------------ |
| Accuracy        | **91.5%**                            |
| F1-score macro  | **0.912**                            |
| Meilleur modèle | `LogisticRegression(C=0.1)`          |
| Vectorisation   | `TfidfVectorizer(ngram_range=(1,2))` |

---

### 2. ✂️ **Résumé de texte (extractif)**

* **Version ML** :

  * TF-IDF vectorisation par phrase
  * Logistic Regression sur chaque phrase (0/1 résumé)
* **Version DL** :

  * Word2Vec + LSTM binaire (chaque phrase = résumé ou non)
  * Prétraitement en séquences indexées avec `Tokenizer`

#### 📊 Résultats du modèle ML (Résumé)

| Paramètre          | Valeur                      |
| ------------------ | --------------------------- |
| Accuracy (phrase)  | **82.6%**                   |
| F1-score (binaire) | **0.831**                   |
| Meilleur modèle    | `LogisticRegression(C=1.0)` |
| Données            | CNN/DailyMail simplifié     |

---

### 3. 🧠 **Extraction de mots-clés**

* **Version ML** :

  * Sélection automatique des mots importants dans une phrase
  * `TfidfVectorizer` + `LogisticRegression`
* **Version DL (optionnelle)** :

  * Embedding + LSTM avec sortie sigmoïde (multi-label)

#### 📊 Résultats du modèle ML (Mots-clés)

| Paramètre       | Valeur                      |
| --------------- | --------------------------- |
| Accuracy        | **88.2%**                   |
| F1-score macro  | **0.879**                   |
| Meilleur modèle | `LogisticRegression(C=0.5)` |

---

### 4. 🌐 **Recherche et génération Wikipedia**

* Extraction de mots-clés
* Requête combinée ou mot-clé par mot-clé via `wikipedia` API
* Résumés récupérés → résumé automatique → corpus contextuel
* Possibilité d’entraîner un modèle à la volée pour sélectionner le meilleur résumé
* Résumé final généré par `Summarizer` (ML ou DL)

---

## 🛠 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/PaulBerdier/DavidLInstruit.git
cd DavidLInstruit
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` installe notamment :

* spaCy + modèles EN/FR
* scikit-learn, pandas, gensim, keras, tensorflow
* `argostranslate`, `wikipedia`, etc.

---

## 🧪 Lancer l'application

### Exécution guidée via `main.py`

```bash
python main.py
```

Menu :

```
📦 CHOISISSEZ UNE OU PLUSIEURS ÉTAPES :
1. 🧠 ENTRAÎNER LES MODÈLES DE CLASSIFICATION
2. ✂️ ENTRAÎNER LES MODÈLES DE RÉSUMÉ
3. 🔑 ENTRAÎNER LE MODÈLE DE MOTS-CLÉS
4. 🌍 TESTER LA TRADUCTION
5. 🚀 LANCER L’INTERFACE (FastAPI)
0. ❌ QUITTER
```

---

## 💻 Structure du projet

```
DavidLInstruit/
│
├── chatbot/
│   ├── interface/                   # Interface FastAPI + HTML/CSS
│   ├── classify.py              # Classifieur (ML et DL)
│   ├── summarize.py             # Résumé de texte
│   ├── keyword_extractor.py     # Mots-clés
│   ├── contextual_builder.py    # Recherche et réponse Wikipedia
│   └── translation.py           # Traduction Argos
│
├── models/                      # Modèles sauvegardés
├── data/                        # Datasets et corpus générés
├── main.py                      # Menu principal
├── requirements.txt             # Dépendances
└── README.md                    # Ce fichier
```

---

## ✍️ Auteur

**Paul Berdier** — Étudiant en M1 Data/IA
Projet académique structuré pour développer un assistant IA multi-tâche intelligent.

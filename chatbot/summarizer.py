import spacy
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Chargement du modèle spaCy français
nlp = spacy.load("fr_core_news_md")

def sentence_similarity(sent1, sent2):
    """Calcule la similarité entre deux phrases à l’aide de leurs vecteurs spaCy"""
    return sent1.vector.reshape(1, -1), sent2.vector.reshape(1, -1)

def build_similarity_matrix(sentences):
    """Construit une matrice de similarité entre phrases"""
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                v1, v2 = sentence_similarity(sentences[i], sentences[j])
                sim_matrix[i][j] = cosine_similarity(v1, v2)[0, 0]
    return sim_matrix

def summarize(text, max_lines=3):
    """Résume un texte en extrayant les phrases les plus centrales"""
    doc = nlp(text)
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 20]

    if len(sentences) <= max_lines:
        return text

    sim_matrix = build_similarity_matrix(sentences)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s.text) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [s for _, s in ranked_sentences[:max_lines]]

    return " ".join(top_sentences)

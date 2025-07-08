# Test du système complet
from src.retrieval.retriever import BaseRetriever

# Créer notre retriever
retriever = BaseRetriever()

# Documents d'exemple - imagine que c'est ta base de connaissances
documents = [
    "Le machine learning est une branche de l'intelligence artificielle.",
    "Les réseaux de neurones sont inspirés du cerveau humain.",
    "Python est le langage le plus populaire pour le machine learning.",
    "Le deep learning utilise des réseaux de neurones profonds.",
    "Les transformers ont révolutionné le traitement du langage naturel.",
    "BERT et GPT sont des modèles de transformers populaires.",
    "La vision par ordinateur permet aux machines de comprendre les images.",
    "Le reinforcement learning apprend par essai et erreur.",
]

# Ajouter des métadonnées (optionnel mais utile)
metadata = [
    {"source": "intro_ml.pdf", "chapter": 1},
    {"source": "neural_networks.pdf", "chapter": 2},
    {"source": "python_guide.pdf", "chapter": 1},
    {"source": "deep_learning.pdf", "chapter": 3},
    {"source": "nlp_transformers.pdf", "chapter": 5},
    {"source": "nlp_transformers.pdf", "chapter": 6},
    {"source": "computer_vision.pdf", "chapter": 1},
    {"source": "rl_basics.pdf", "chapter": 1},
]

# Indexer les documents
retriever.index_documents(documents, metadata)

# Tester quelques requêtes
queries = [
    "Qu'est-ce que les transformers ?",
    "Comment fonctionne le deep learning ?",
    "Python et machine learning"
]

for query in queries:
    print(f"\n=== Requête : {query} ===")
    results = retriever.retrieve(query, k=3)

    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.3f}")
        print(f"   Texte: {result['text']}")
        print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
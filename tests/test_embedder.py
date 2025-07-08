from src.retrieval.embedder import DocumentEmbedder

# Initialisons notre embedder
embedder = DocumentEmbedder()

# Test 1 : Embeddings de phrases similaires
print("=== Test 1 : Phrases similaires ===")
phrases_similaires = [
    "Le chat mange sa nourriture",
    "Le félin consomme son repas",
    "Le chien mange ses croquettes"
]

# Créons les embeddings
embeddings = embedder.embed_text(phrases_similaires)
print(f"Shape des embeddings : {embeddings.shape}")

# Calculons les similarités
for i in range(len(phrases_similaires)):
    for j in range(i+1, len(phrases_similaires)):
        sim = embedder.compute_similarity(embeddings[i], embeddings[j])
        print(f"Similarité entre '{phrases_similaires[i]}' et '{phrases_similaires[j]}': {sim:.3f}")

# Test 2 : Phrases très différentes
print("\n=== Test 2 : Phrases différentes ===")
phrases_differentes = [
    "Le machine learning révolutionne l'informatique",
    "J'aime manger des pommes au petit déjeuner",
    "La physique quantique est complexe"
]

embeddings_diff = embedder.embed_text(phrases_differentes)
for i in range(len(phrases_differentes)):
    for j in range(i+1, len(phrases_differentes)):
        sim = embedder.compute_similarity(embeddings_diff[i], embeddings_diff[j])
        print(f"Similarité entre phrases différentes : {sim:.3f}")
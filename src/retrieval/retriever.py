from typing import List, Dict, Optional

from .embedder import DocumentEmbedder
from .vector_store import VectorStore


class BaseRetriever:
    """
    Le retriever coordonne l'embedder et le vector store pour :
    1. Indexer des documents
    2. Rechercher les plus pertinents pour une requête
    """

    def __init__(self, embedder: Optional[DocumentEmbedder] = None):
        # Utiliser l'embedder fourni ou en créer un nouveau
        self.embedder = embedder or DocumentEmbedder()

        # Créer le vector store avec la bonne dimension
        self.vector_store = VectorStore(self.embedder.embedding_dim)

    def index_documents(self,
                        documents: List[str],
                        metadata: Optional[List[Dict]] = None,
                        batch_size: int = 32):
        """
        Indexe une liste de documents.

        C'est ici que la magie opère :
        1. On transforme le texte en embeddings
        2. On les stocke dans le vector store
        """
        print(f"Indexation de {len(documents)} documents...")

        # Créer les embeddings par batch pour l'efficacité
        embeddings = self.embedder.embed_text(documents, batch_size=batch_size)

        # Ajouter au store
        self.vector_store.add_documents(documents, embeddings, metadata)

        print("Indexation terminée !")

    def retrieve(self,
                 query: str,
                 k: int = 5) -> List[Dict[str, any]]:
        """
        Récupère les k documents les plus pertinents pour la requête.

        Returns:
            Liste de dictionnaires avec les infos de chaque document
        """
        # Créer l'embedding de la requête
        query_embedding = self.embedder.embed_text(query)[0]

        # Rechercher
        results = self.vector_store.search(query_embedding, k)

        # Formatter les résultats
        formatted_results = []
        for idx, distance, text, metadata in results:
            # Convertir la distance en score de similarité
            # Pour L2, plus c'est petit, plus c'est similaire
            similarity_score = 1 / (1 + distance)

            formatted_results.append({
                "index": idx,
                "text": text,
                "score": similarity_score,
                "distance": distance,
                "metadata": metadata
            })

        return formatted_results
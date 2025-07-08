import os
import pickle
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np


class VectorStore:
    """
    Cette classe gère le stockage et la recherche efficace d'embeddings.

    Pourquoi utiliser Faiss ?
    - Développé par Facebook AI Research
    - Ultra-rapide pour la recherche de similarité
    - Peut gérer des millions de vecteurs
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Args:
            embedding_dim: Dimension des vecteurs (doit correspondre à l'embedder)
            index_type: Type d'index Faiss
                - "flat" : Exact mais lent pour beaucoup de données
                - "ivf" : Approximatif mais rapide
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Créer l'index Faiss
        if index_type == "flat":
            # IndexFlatL2 = recherche exacte avec distance L2
            # On pourrait aussi utiliser IndexFlatIP pour le produit scalaire
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Pour l'instant, on reste simple
            raise NotImplementedError("Seul 'flat' est supporté pour l'instant")

        # Stocker les métadonnées (textes originaux, sources, etc.)
        self.documents = []
        self.metadata = []

    def add_documents(self,
                      texts: List[str],
                      embeddings: np.ndarray,
                      metadata: Optional[List[Dict]] = None):
        """
        Ajoute des documents au store.

        Args:
            texts: Les textes originaux
            embeddings: Les embeddings correspondants
            metadata: Info supplémentaire (source, date, etc.)
        """
        # Vérifications de sécurité
        assert len(texts) == embeddings.shape[0], "Nombre de textes != nombre d'embeddings"
        assert embeddings.shape[
                   1] == self.embedding_dim, f"Dimension incorrecte : {embeddings.shape[1]} != {self.embedding_dim}"

        # Ajouter à l'index Faiss
        self.index.add(embeddings.astype('float32'))

        # Ajouter aux métadonnées
        self.documents.extend(texts)

        if metadata is None:
            metadata = [{} for _ in texts]
        self.metadata.extend(metadata)

        print(f"Ajouté {len(texts)} documents. Total : {len(self.documents)}")

    def search(self,
               query_embedding: np.ndarray,
               k: int = 5) -> List[Tuple[int, float, str, Dict]]:
        """
        Recherche les k documents les plus similaires.

        Returns:
            Liste de tuples (index, distance, texte, metadata)
        """
        # Faiss veut du float32 et shape (n_queries, dim)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Recherche - D = distances, I = indices
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx < len(self.documents):  # Vérification de sécurité
                results.append((
                    idx,
                    distances[0][i],
                    self.documents[idx],
                    self.metadata[idx]
                ))

        return results

    def save(self, path: str):
        """Sauvegarde le store sur disque."""
        os.makedirs(path, exist_ok=True)

        # Sauvegarder l'index Faiss
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Sauvegarder les métadonnées
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim
            }, f)

        print(f"Store sauvegardé dans {path}")

    def load(self, path: str):
        """Charge le store depuis le disque."""
        # Charger l'index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        # Charger les métadonnées
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.embedding_dim = data["embedding_dim"]

        print(f"Store chargé depuis {path}. {len(self.documents)} documents.")
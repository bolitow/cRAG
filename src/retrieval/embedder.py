import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch


class DocumentEmbedder:
    """
    Cette classe transforme des textes en vecteurs numériques (embeddings).

    Pourquoi c'est important ?
    - Les ordinateurs ne comprennent pas le texte directement
    - Les embeddings capturent le sens sémantique du texte
    - On peut calculer la similarité entre deux embeddings facilement
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialise l'embedder avec un modèle pré-entraîné.

        all-MiniLM-L6-v2 est un bon choix car :
        - Rapide (seulement 80MB)
        - Bon équilibre performance/vitesse
        - Entraîné sur 1 milliard de paires de phrases
        """
        print(f"Chargement du modèle d'embedding : {model_name}")
        self.model = SentenceTransformer(model_name)

        # Vérifions si on peut utiliser CUDA pour accélérer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Modèle chargé sur : {self.device}")

        # Dimension des embeddings (important pour la suite)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Dimension des embeddings : {self.embedding_dim}")

    def embed_text(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Transforme un texte ou une liste de textes en embeddings.

        Args:
            text: Un texte seul ou une liste de textes
            batch_size: Nombre de textes à traiter en même temps (pour l'efficacité)

        Returns:
            numpy array de shape (n_texts, embedding_dim)
        """
        # Si c'est un seul texte, on le met dans une liste
        if isinstance(text, str):
            text = [text]

        # Le modèle fait tout le travail complexe pour nous
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=len(text) > 100,  # Barre de progression si beaucoup de textes
            convert_to_numpy=True
        )

        return embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings.

        La similarité cosinus mesure l'angle entre deux vecteurs :
        - 1.0 = identiques
        - 0.0 = orthogonaux (aucun rapport)
        - -1.0 = opposés

        En pratique, pour du texte, on a souvent des valeurs entre 0 et 1.
        """
        # Normalisation des vecteurs
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)

        # Produit scalaire des vecteurs normalisés = cosinus de l'angle
        similarity = np.dot(norm1, norm2)

        return float(similarity)
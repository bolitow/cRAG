# Test complet du pipeline CRAG
import sys

sys.path.append('../src')

from pipeline.crag_pipeline import CRAGPipeline

# Créer le pipeline complet
print("🚀 Initialisation du pipeline CRAG complet...")
pipeline = CRAGPipeline(verbose=True)

# Documents de test - Une base de connaissances sur les transformers
documents = [
    """Les transformers sont une architecture de réseau de neurones révolutionnaire introduite en 2017 
    dans le papier "Attention is All You Need". Ils ont transformé le domaine du traitement du langage naturel
    en permettant un traitement parallèle efficace des séquences, contrairement aux RNN qui traitent
    les données séquentiellement.""",

    """Le mécanisme d'attention est le cœur des transformers. Il calcule des scores d'attention entre
    tous les éléments d'une séquence, permettant au modèle de capturer des dépendances à long terme.
    L'attention multi-têtes divise la représentation en plusieurs sous-espaces, permettant au modèle
    d'apprendre différents types de relations simultanément.""",

    """Les composants principaux d'un transformer incluent :
    - L'encodeur : transforme la séquence d'entrée en représentations
    - Le décodeur : génère la séquence de sortie
    - Les couches d'attention multi-têtes
    - Les réseaux feed-forward
    - La normalisation par couches (Layer Norm)
    - Les connexions résiduelles
    - L'encodage positionnel pour capturer l'ordre des mots""",

    """BERT (Bidirectional Encoder Representations from Transformers) utilise uniquement la partie
    encodeur des transformers. Il est pré-entraîné sur deux tâches : prédiction de mots masqués
    et prédiction de la phrase suivante. Cette approche bidirectionnelle lui permet de comprendre
    le contexte complet d'un mot.""",

    """GPT (Generative Pre-trained Transformer) utilise uniquement la partie décodeur des transformers.
    Il est entraîné de manière autoregressive, prédisant le mot suivant étant donné tous les mots
    précédents. Cette approche le rend excellent pour la génération de texte.""",

    """Les transformers ont révolutionné de nombreux domaines au-delà du NLP. Vision Transformer (ViT)
    applique l'architecture aux images en les découpant en patches. DALL-E utilise les transformers
    pour générer des images à partir de descriptions textuelles. Les transformers sont maintenant
    utilisés en biologie, musique, et même en chimie."""
]

# Métadonnées pour tracer l'origine
metadata = [
    {"source": "introduction_transformers.pdf", "section": "Overview"},
    {"source": "attention_mechanism.pdf", "section": "Core Concepts"},
    {"source": "transformer_architecture.pdf", "section": "Components"},
    {"source": "bert_paper.pdf", "section": "Model Description"},
    {"source": "gpt_paper.pdf", "section": "Architecture"},
    {"source": "transformers_applications.pdf", "section": "Beyond NLP"}
]

# Indexer les documents
print("\n📚 Indexation de la base de connaissances...")
pipeline.index_documents(documents, metadata)

# Tester plusieurs questions
questions = [
    "Comment fonctionnent les transformers ?",
    "Qu'est-ce que BERT ?",
    "Quelle est la différence entre BERT et GPT ?",
    "Pourquoi les transformers sont-ils révolutionnaires ?",
    "Quels sont les composants d'un transformer ?"
]

print("\n🎯 Test du pipeline avec différentes questions...\n")

for i, question in enumerate(questions, 1):
    print(f"\n{'#' * 80}")
    print(f"QUESTION {i}/{len(questions)}")
    print(f"{'#' * 80}")

    # Traiter la question
    result = pipeline.process_query(question, debug=True)

    # Afficher la réponse
    print(f"\n{'=' * 60}")
    print("📝 RÉPONSE GÉNÉRÉE")
    print(f"{'=' * 60}")
    print(result.answer)

    # Afficher les métadonnées
    print(f"\n{'=' * 60}")
    print("📊 MÉTADONNÉES")
    print(f"{'=' * 60}")
    print(f"Confiance : {result.confidence:.2%}")
    print(f"Temps de traitement : {result.processing_time:.2f}s")
    print(f"Besoin d'infos supplémentaires : {'Oui' if result.needs_more_info else 'Non'}")

    # Si mode debug, afficher plus de détails
    if result.debug_info:
        print(f"\n🔍 DEBUG - Top strips utilisés :")
        for j, strip in enumerate(result.debug_info["top_graded_strips"][:3], 1):
            print(f"   {j}. [{strip['category']}] {strip['content']}")

    # Pause entre les questions
    if i < len(questions):
        input(f"\n>>> Appuyez sur Entrée pour la question suivante...")

# Résumé final
print(f"\n{'=' * 80}")
print("🏁 TEST TERMINÉ")
print(f"{'=' * 80}")
print(f"Questions traitées : {len(questions)}")
print("Le pipeline CRAG est opérationnel ! 🎉")
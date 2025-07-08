# tests/test_crag_with_llm.py
import sys

sys.path.append('../src')

from pipeline.crag_pipeline import CRAGPipeline
from generation.response_generator import ResponseGenerator

# Créer le générateur avec ton endpoint LLM
generator = ResponseGenerator(
    llm_endpoint="http://141.94.106.229:8085/generate",
    use_llm=True,
    use_templates=False,  # Désactiver les templates pour utiliser le LLM
    response_style="educational"
)

# Créer le pipeline avec le générateur LLM
print("🚀 Initialisation du pipeline CRAG avec LLM...")
pipeline = CRAGPipeline(
    generator=generator,
    verbose=True
)

# Documents de test
documents = [
    """Les transformers sont une architecture de réseau de neurones révolutionnaire introduite en 2017.
    Ils utilisent le mécanisme d'attention pour traiter les séquences de manière parallèle,
    contrairement aux RNN qui traitent les données séquentiellement.""",

    """Le mécanisme d'attention est le cœur des transformers. Il calcule des scores d'attention
    entre tous les éléments d'une séquence. L'attention multi-têtes permet au modèle
    d'apprendre différents types de relations simultanément.""",

    """BERT (Bidirectional Encoder Representations from Transformers) utilise uniquement
    la partie encodeur. Il est pré-entraîné sur la prédiction de mots masqués et comprend
    le contexte bidirectionnel complet."""
]

# Indexer les documents
pipeline.index_documents(documents)

# Tester une question
question = "Comment fonctionnent les transformers ?"
print(f"\n🔍 Question : {question}")

# Traiter la question
result = pipeline.process_query(question)

# Afficher la réponse
print(f"\n📝 RÉPONSE GÉNÉRÉE PAR LLM :")
print("=" * 60)
print(result.answer)
print("=" * 60)
print(f"\nMéthode utilisée : {result.steps_details['generation']['method']}")
print(f"Confiance : {result.confidence:.2%}")
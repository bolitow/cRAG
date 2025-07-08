# tests/test_relevance_grader.py
from src.retrieval.retriever import BaseRetriever
from src.retrieval.embedder import DocumentEmbedder
from src.grading.knowledge_stripper import KnowledgeStripper
from src.grading.relevance_grader import RelevanceGrader

# Initialiser tous nos composants
embedder = DocumentEmbedder()
retriever = BaseRetriever(embedder)
stripper = KnowledgeStripper()
grader = RelevanceGrader(
    model_name="google/gemma-3-4b-it",  # ou ton modèle local
    use_llm=False,  # On commence sans LLM pour tester
    embedder=embedder
)

# Documents de test plus riches
documents = [
    """Les transformers sont une architecture de réseau de neurones introduite en 2017. 
    Ils utilisent le mécanisme d'attention pour traiter les séquences. 
    L'attention permet au modèle de se concentrer sur différentes parties de l'entrée.
    Les transformers ont révolutionné le NLP grâce à leur capacité de parallélisation.""",

    """BERT (Bidirectional Encoder Representations from Transformers) est un modèle pré-entraîné.
    Il utilise l'architecture transformer pour comprendre le contexte bidirectionnel.
    BERT a été entraîné sur Wikipedia et BookCorpus.
    Il excelle dans les tâches de compréhension du langage.""",

    """Les réseaux de neurones convolutifs (CNN) sont utilisés en vision par ordinateur.
    Ils appliquent des filtres pour détecter des motifs dans les images.
    Les CNN ont plusieurs couches : convolution, pooling, et fully connected.
    Ils sont très différents des transformers."""
]

# Indexer les documents
retriever.index_documents(documents)

# Requête test
query = "Comment fonctionnent les transformers ?"

# 1. Récupérer les documents pertinents
print("=== ÉTAPE 1 : RETRIEVAL ===")
retrieved_docs = retriever.retrieve(query, k=2)
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i + 1} (score: {doc['score']:.3f}):")
    print(doc['text'][:100] + "...")

# 2. Découper en knowledge strips
print("\n\n=== ÉTAPE 2 : KNOWLEDGE STRIPPING ===")
all_strips = []
for doc in retrieved_docs:
    strips = stripper.strip_document(doc['text'], doc['index'])
    all_strips.extend(strips)
    print(f"Document {doc['index']} découpé en {len(strips)} strips")

# 3. Grader les strips
print("\n\n=== ÉTAPE 3 : RELEVANCE GRADING ===")
graded_strips = grader.grade_strips(query, all_strips)

# Afficher les résultats
print(f"\nStrips triés par pertinence :")
print("-" * 80)

for i, graded in enumerate(graded_strips[:5]):  # Top 5
    print(f"\n{i + 1}. {graded.relevance_category.name} (score: {graded.relevance_score:.3f})")
    print(f"   Contenu : {graded.strip.content[:100]}...")
    print(f"   Type : {graded.strip.strip_type}")
    print(f"   Raisonnement : {graded.reasoning}")
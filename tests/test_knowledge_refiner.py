from src.retrieval.retriever import BaseRetriever
from src.retrieval.embedder import DocumentEmbedder
from src.grading.knowledge_stripper import KnowledgeStripper
from src.grading.relevance_grader import RelevanceGrader
from src.refinement.knowledge_refiner import KnowledgeRefiner

# Initialisation de tous les composants
embedder = DocumentEmbedder()
retriever = BaseRetriever(embedder)
stripper = KnowledgeStripper()
grader = RelevanceGrader(model_name='google/gemma-3-4b-it', use_llm=True, embedder=embedder)
# Dans test_knowledge_refiner.py
refiner = KnowledgeRefiner(
    relevance_threshold=0.5,
    similarity_threshold=0.85,
    max_strips=7,
    min_coverage=0.7,
    embedder=embedder
)

# Documents de test enrichis
documents = [
    """Les transformers sont une architecture de réseau de neurones introduite en 2017.
    Ils utilisent le mécanisme d'attention pour traiter les séquences.
    L'attention permet au modèle de se concentrer sur différentes parties de l'entrée.
    Cette architecture a révolutionné le traitement du langage naturel.""",

    """L'attention est le mécanisme clé des transformers. Elle calcule des scores
    entre tous les éléments d'une séquence. Les transformers utilisent l'attention
    multi-têtes pour capturer différents types de relations. C'est ce qui leur
    permet de comprendre le contexte global.""",

    """BERT utilise l'architecture transformer de manière bidirectionnelle.
    GPT utilise les transformers de manière autoregressive.
    Les deux modèles ont montré des performances exceptionnelles.
    Les transformers permettent le traitement parallèle des séquences.""",

    """Les composants principaux des transformers incluent :
    - Les couches d'attention multi-têtes
    - Les réseaux feed-forward
    - La normalisation par couches
    - Les connexions résiduelles
    Ces éléments travaillent ensemble pour traiter l'information.""",

    # Ajoutons du bruit
    """Les réseaux de neurones convolutifs sont excellents pour la vision.
    Ils utilisent des filtres pour détecter des caractéristiques.
    Les CNN ont dominé la vision par ordinateur pendant des années."""
]

# Indexer les documents
retriever.index_documents(documents)

# Requête test
query = "Comment fonctionnent les transformers ?"

print("=== PIPELINE CRAG COMPLET ===\n")

# Étape 1 : Retrieval
print("1️⃣ RETRIEVAL")
retrieved_docs = retriever.retrieve(query, k=5)
print(f"   Documents récupérés : {len(retrieved_docs)}")

# Étape 2 : Knowledge Stripping
print("\n2️⃣ KNOWLEDGE STRIPPING")
all_strips = []
for doc in retrieved_docs:
    strips = stripper.strip_document(doc['text'], doc['index'])
    all_strips.extend(strips)
print(f"   Strips extraits : {len(all_strips)}")

# Étape 3 : Relevance Grading
print("\n3️⃣ RELEVANCE GRADING")
graded_strips = grader.grade_strips(query, all_strips)
print(f"   Strips gradés :")
for i, gs in enumerate(graded_strips[:5]):
    print(f"   - {gs.relevance_category.name}: {gs.strip.content[:50]}...")

# Étape 4 : Knowledge Refinement
print("\n4️⃣ KNOWLEDGE REFINEMENT")
refined_knowledge = refiner.refine_knowledge(graded_strips, query)

print(f"\n📊 RÉSULTATS DU RAFFINEMENT :")
print(f"   {refined_knowledge.summary}")

print(f"\n📋 STRIPS FINAUX ORGANISÉS :")
for i, strip in enumerate(refined_knowledge.strips):
    print(f"\n   {i + 1}. [{strip.strip.strip_type}] Score: {strip.relevance_score:.3f}")
    print(f"      {strip.strip.content[:80]}...")

if refined_knowledge.missing_aspects:
    print(f"\n⚠️  ASPECTS MANQUANTS : {', '.join(refined_knowledge.missing_aspects)}")

if refined_knowledge.needs_additional_search:
    print("\n🔍 RECOMMANDATION : Effectuer une recherche supplémentaire")
    print("   Raisons possibles :")
    print("   - Couverture insuffisante des aspects requis")
    print("   - Manque d'informations sur le mécanisme détaillé")
else:
    print("\n✅ CONTEXTE SUFFISANT : Prêt pour la génération de réponse")

# Afficher le contexte final qui serait envoyé au LLM
print("\n📝 CONTEXTE POUR LA GÉNÉRATION :")
print("-" * 50)
context = "\n\n".join([strip.strip.content for strip in refined_knowledge.strips])
print(context[:500] + "..." if len(context) > 500 else context)
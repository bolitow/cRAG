# tests/test_knowledge_stripper.py
from src.grading.knowledge_stripper import KnowledgeStripper

# Créer notre stripper
stripper = KnowledgeStripper()

# Document de test avec différents types de contenu
test_document = """
Les transformers sont une architecture de réseau de neurones révolutionnaire. Ils ont été introduits 
en 2017 dans le papier "Attention is All You Need". Cette architecture utilise le mécanisme d'attention 
pour traiter les séquences de manière parallèle.

Le principal avantage des transformers est leur capacité à capturer les dépendances à long terme. 
Par exemple, BERT peut comprendre le contexte d'un mot en regardant tous les autres mots de la phrase. 
GPT, quant à lui, génère du texte de manière autoregressive.

Les composants clés des transformers incluent :
- L'attention multi-têtes
- Les couches de feed-forward
- La normalisation par couches
- Les embeddings positionnels

Cette architecture a révolutionné de nombreux domaines comme la traduction automatique, 
la génération de texte et même la vision par ordinateur.
"""

# Tester le stripping
strips = stripper.strip_document(test_document, doc_id=0, granularity="adaptive")

print(f"Nombre de strips extraits : {len(strips)}\n")

for i, strip in enumerate(strips):
    print(f"Strip {i+1} ({strip.strip_type}):")
    print(f"Contenu : {strip.content[:100]}...")
    print(f"Contexte : {strip.context}")
    print("-" * 50)
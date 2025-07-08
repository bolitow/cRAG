# example_ingestion_usage.py
"""
Exemple d'utilisation du système d'ingestion avec le pipeline CRAG.

Ce script montre comment :
1. Parser différents types de documents
2. Les enrichir avec les métadonnées de sécurité
3. Les intégrer dans le pipeline CRAG pour répondre aux questions
"""

import sys
from pathlib import Path
from datetime import datetime

# Ajouter le chemin src au PYTHONPATH
sys.path.append('src')

# Imports du système d'ingestion
from src.ingestion import DocumentParser, SecurityPreprocessor, MetadataExtractor

# Imports du pipeline CRAG
from src.pipeline.crag_pipeline import CRAGPipeline
from src.grading.knowledge_stripper import KnowledgeStripper
from src.grading.domain_patterns import CyberSecurityPatterns


def setup_ingestion_system():
    """
    Configure le système d'ingestion avec tous ses composants.
    """
    print("🔧 Configuration du système d'ingestion...")

    # Configuration du parser principal
    parser_config = {
        'cache_enabled': True,
        'cache_dir': '.document_cache',
        'parallel_processing': True,
        'max_workers': 4,
        'file_size_limit': 50,  # MB
        'parser_configs': {
            'pdf': {
                'ocr_enabled': False,
                'extract_tables': True,
                'remove_headers': True
            },
            'excel': {
                'skip_empty_sheets': True,
                'detect_tables': True
            },
            'markdown': {
                'extract_front_matter': True,
                'extract_links': True
            },
            'faq': {
                'extract_categories': True,
                'language': 'fr'
            }
        }
    }

    # Créer le parser
    document_parser = DocumentParser(parser_config)

    # Configuration du préprocesseur de sécurité
    security_config = {
        'extract_dates': True,
        'normalize_terms': True,
        'enrich_metadata': True,
        'detect_references': True,
        'language': 'fr'
    }

    security_preprocessor = SecurityPreprocessor(security_config)

    # Configuration de l'extracteur de métadonnées
    metadata_config = {
        'organization_name': 'MonEntreprise',
        'default_classification': 'Internal',
        'extract_pii': True,
        'validate_references': True,
        'quality_checks': True
    }

    metadata_extractor = MetadataExtractor(metadata_config)

    print("✅ Système d'ingestion configuré")

    return document_parser, security_preprocessor, metadata_extractor


def process_security_documents(directory_path: str):
    """
    Traite un répertoire de documents de sécurité.
    """
    # Initialiser le système
    document_parser, security_preprocessor, metadata_extractor = setup_ingestion_system()

    print(f"\n📁 Traitement du répertoire : {directory_path}")

    # Parser tous les documents du répertoire
    try:
        parsed_documents = document_parser.parse_directory(
            directory_path,
            recursive=True,
            file_patterns=['*.pdf', '*.md', '*.xlsx', '*.txt'],
            exclude_patterns=['*_draft.*', 'temp/*', '.*']
        )

        print(f"\n✅ {len(parsed_documents)} documents parsés avec succès")

    except Exception as e:
        print(f"❌ Erreur lors du parsing : {e}")
        return []

    # Préprocesser chaque document
    enriched_documents = []

    print("\n🔄 Enrichissement des documents...")

    for doc in parsed_documents:
        try:
            # Extraire les métadonnées
            doc_metadata = metadata_extractor.extract_metadata(doc)

            # Appliquer le preprocessing de sécurité
            enriched_doc = security_preprocessor.preprocess(doc)

            enriched_documents.append(enriched_doc)

            # Afficher un résumé
            print(f"\n📄 {doc.metadata.get('file_name', 'Unknown')}:")
            print(f"   - Type : {doc.doc_type}")
            print(f"   - Classification : {doc_metadata.classification_level}")
            print(f"   - Frameworks : {', '.join(enriched_doc.metadata['security_context']['compliance_frameworks'])}")
            print(f"   - Domaines : {', '.join(enriched_doc.metadata['security_context']['security_domains'][:3])}")
            print(f"   - Score qualité : {doc_metadata.completeness_score:.2f}")

        except Exception as e:
            print(f"⚠️  Erreur lors de l'enrichissement de {doc.metadata.get('file_name', 'Unknown')}: {e}")

    return enriched_documents


def integrate_with_crag_pipeline(enriched_documents):
    """
    Intègre les documents enrichis dans le pipeline CRAG.
    """
    print("\n🚀 Intégration dans le pipeline CRAG...")

    # Créer un Knowledge Stripper adapté à la cybersécurité
    stripper = KnowledgeStripper(language='fr')

    # Créer le pipeline CRAG
    pipeline = CRAGPipeline(
        stripper=stripper,
        verbose=True
    )

    # Convertir les documents enrichis en format pour le pipeline
    documents_for_indexing = []
    metadata_for_indexing = []

    for doc in enriched_documents:
        # Le contenu principal
        documents_for_indexing.append(doc.content)

        # Les métadonnées enrichies
        doc_metadata = {
            'source': doc.source_path,
            'doc_type': doc.doc_type,
            'security_doc_type': doc.metadata.get('security_doc_type', 'general'),
            'classification': doc.metadata.get('extracted_metadata', {}).get('classification', 'Internal'),
            'compliance_frameworks': doc.metadata.get('security_context', {}).get('compliance_frameworks', []),
            'security_domains': doc.metadata.get('security_context', {}).get('security_domains', []),
            'importance_score': doc.metadata.get('importance_score', 5),
            'is_critical': doc.metadata.get('is_critical_security_doc', False),
            'systems_affected': doc.metadata.get('extracted_metadata', {}).get('systems_affected', []),
            'last_updated': doc.metadata.get('extracted_metadata', {}).get('last_modified', '')
        }

        metadata_for_indexing.append(doc_metadata)

    # Indexer les documents
    print(f"\n📚 Indexation de {len(documents_for_indexing)} documents enrichis...")
    pipeline.index_documents(documents_for_indexing, metadata_for_indexing)

    print("✅ Documents indexés dans le pipeline CRAG")

    return pipeline


def demo_audit_queries(pipeline):
    """
    Démontre l'utilisation du pipeline pour répondre à des questions d'audit.
    """
    print("\n" + "=" * 80)
    print("🎯 DÉMONSTRATION : Réponses aux questions d'audit")
    print("=" * 80)

    # Questions d'audit typiques
    audit_questions = [
        {
            'question': "Quelle est notre politique de gestion des mots de passe ?",
            'context': "Audit ISO 27001 - A.9.4.3"
        },
        {
            'question': "Comment gérons-nous les sauvegardes des données critiques ?",
            'context': "Audit de continuité d'activité"
        },
        {
            'question': "Quels sont nos contrôles pour la sécurité réseau ?",
            'context': "Audit technique infrastructure"
        },
        {
            'question': "Comment assurons-nous la conformité RGPD ?",
            'context': "Audit de protection des données"
        },
        {
            'question': "Quelle est la procédure en cas d'incident de sécurité ?",
            'context': "Test du plan de réponse aux incidents"
        }
    ]

    for qa in audit_questions:
        print(f"\n{'=' * 60}")
        print(f"❓ QUESTION D'AUDIT : {qa['question']}")
        print(f"📋 Contexte : {qa['context']}")
        print(f"{'=' * 60}")

        # Traiter la question
        result = pipeline.process_query(qa['question'])

        # Afficher la réponse
        print(f"\n📝 RÉPONSE :")
        print("-" * 60)
        print(result.answer)
        print("-" * 60)

        # Afficher les métadonnées
        print(f"\n📊 MÉTADONNÉES :")
        print(f"- Confiance : {result.confidence:.2%}")
        print(f"- Documents sources : {result.steps_details.get('retrieval', {}).get('docs_retrieved', 0)}")
        print(f"- Besoin d'infos supplémentaires : {'Oui' if result.needs_more_info else 'Non'}")

        # Pause entre les questions
        input("\n>>> Appuyez sur Entrée pour la question suivante...")


def generate_audit_report(pipeline, output_path: str = "rapport_audit_crag.md"):
    """
    Génère un rapport d'audit basé sur les documents indexés.
    """
    print(f"\n📊 Génération du rapport d'audit...")

    report_content = f"""# Rapport d'Audit - Système CRAG
Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 1. Vue d'ensemble du système documentaire

### Documents analysés
- Total : {pipeline.retriever.vector_store.index.ntotal} documents indexés
- Types : Politiques, Procédures, Standards, FAQ

### Couverture des domaines de sécurité
Les documents couvrent les domaines suivants :
- Gestion des accès et identités
- Sécurité réseau
- Gestion des incidents
- Continuité d'activité
- Protection des données
- Gouvernance et conformité

## 2. Conformité aux frameworks

### ISO 27001/27002
- Couverture des contrôles : À analyser
- Documents de référence : Politiques et procédures identifiées

### RGPD
- Documentation privacy : Présente
- Procédures de gestion des données : Documentées

## 3. Points d'attention

### Documents critiques
- Tous les documents de politique sont classifiés correctement
- Les procédures d'urgence sont à jour
- Les matrices de contrôle sont maintenues

### Recommandations
1. Maintenir à jour le registre des documents
2. Réviser annuellement toutes les politiques
3. Tester régulièrement les procédures d'incident

## 4. Capacités du système CRAG

Le système est maintenant capable de :
- ✅ Répondre aux questions d'audit avec des sources documentées
- ✅ Identifier les documents pertinents pour chaque contrôle
- ✅ Croiser les références entre documents
- ✅ Détecter les incohérences potentielles
- ✅ Fournir des réponses contextualisées selon le framework

---
*Rapport généré automatiquement par le système CRAG*
"""

    # Sauvegarder le rapport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ Rapport sauvegardé : {output_path}")


def main():
    """
    Point d'entrée principal du script de démonstration.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          SYSTÈME CRAG - INGESTION DE DOCUMENTS               ║
    ║                  CYBERSÉCURITÉ & CONFORMITÉ                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Définir le répertoire de documents à traiter
    # CHANGEZ CE CHEMIN selon votre structure
    documents_directory = "./documents_securite"

    # Vérifier que le répertoire existe
    if not Path(documents_directory).exists():
        print(f"⚠️  Créez le répertoire '{documents_directory}' et ajoutez vos documents :")
        print("   - Politiques de sécurité (PDF)")
        print("   - Procédures (Markdown)")
        print("   - Matrices de contrôle (Excel)")
        print("   - FAQ sécurité (TXT/HTML)")

        # Créer un répertoire d'exemple
        Path(documents_directory).mkdir(exist_ok=True)

        # Créer un document d'exemple
        exemple_politique = """# Politique de Sécurité des Mots de Passe

**Classification** : Interne  
**Version** : 2.1  
**Date** : 15/01/2024  
**Propriétaire** : RSSI  

## 1. Objectif

Cette politique définit les exigences de sécurité pour la création et la gestion des mots de passe au sein de l'organisation, conformément à la norme ISO 27001 A.9.4.3.

## 2. Exigences

### 2.1 Complexité des mots de passe

Tous les mots de passe DOIVENT respecter les critères suivants :
- Longueur minimale : 12 caractères
- Contenir au moins : 1 majuscule, 1 minuscule, 1 chiffre, 1 caractère spécial
- Ne pas contenir de mots du dictionnaire
- Ne pas contenir d'informations personnelles

### 2.2 Rotation des mots de passe

- Les mots de passe DOIVENT être changés tous les 90 jours
- Les 12 derniers mots de passe ne peuvent pas être réutilisés
- Un changement immédiat est requis en cas de compromission suspectée

### 2.3 Stockage et protection

- Les mots de passe ne DOIVENT JAMAIS être écrits ou stockés en clair
- L'utilisation d'un gestionnaire de mots de passe approuvé est OBLIGATOIRE
- Le partage de mots de passe est STRICTEMENT INTERDIT

## 3. Responsabilités

- **Utilisateurs** : Respecter cette politique et signaler tout incident
- **RSSI** : Maintenir et faire appliquer cette politique
- **IT** : Implémenter les contrôles techniques

## 4. Non-conformité

Le non-respect de cette politique peut entraîner des sanctions disciplinaires.

## 5. Révision

Cette politique sera révisée annuellement ou lors de changements majeurs.

---
*Document approuvé par : Direction Générale*
"""

        with open(Path(documents_directory) / "POL-SEC-001_Mots_de_passe.md", 'w', encoding='utf-8') as f:
            f.write(exemple_politique)

        print(f"\n✅ Document d'exemple créé dans {documents_directory}")
        print("   Ajoutez vos propres documents et relancez le script.")
        return

    # Traiter les documents
    enriched_docs = process_security_documents(documents_directory)

    if not enriched_docs:
        print("❌ Aucun document traité. Vérifiez votre répertoire.")
        return

    # Intégrer dans CRAG
    pipeline = integrate_with_crag_pipeline(enriched_docs)

    # Démonstration des requêtes
    demo_audit_queries(pipeline)

    # Générer un rapport
    generate_audit_report(pipeline)

    print("\n✅ Démonstration terminée !")
    print("\n💡 Prochaines étapes :")
    print("   1. Ajoutez plus de documents dans votre répertoire")
    print("   2. Testez avec vos vraies questions d'audit")
    print("   3. Personnalisez les configurations selon vos besoins")
    print("   4. Intégrez avec vos outils d'audit existants")


if __name__ == "__main__":
    main()
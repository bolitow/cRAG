# tests/test_audit_grading.py
"""
Test du système de grading amélioré pour les audits.

Ce script démontre comment les nouvelles règles d'audit améliorent
la sélection des documents pertinents pour répondre aux auditeurs.
"""

import sys

sys.path.append('../src')

from src.grading.knowledge_stripper import KnowledgeStripper
from src.grading.relevance_grader import RelevanceGrader
from src.grading.audit_relevance_rules import AuditContext


def create_test_strips():
    """
    Crée des strips de test représentant différents types de documents.
    """
    from src.grading.knowledge_stripper import KnowledgeStrip

    strips = [
        # Strip 1 : Politique officielle ISO 27001
        KnowledgeStrip(
            content="""La politique de gestion des accès de notre organisation établit que tous les accès 
            aux systèmes d'information doivent être accordés selon le principe du moindre privilège. 
            Cette politique est conforme à l'exigence ISO 27001 A.9.1 et fait l'objet d'une revue 
            annuelle par la direction. Dernière mise à jour : janvier 2024.""",
            strip_type="policy_statement",
            position=0,
            
            source_doc_id=1,
            domain_category="policy",
            compliance_refs=["ISO 27001 A.9.1"],
            context={
                'doc_type': 'policy',
                'last_updated': '2024-01-15',
                'is_official': True,
                'version': '3.0'
            }
        ),

        # Strip 2 : Procédure technique mais ancienne
        KnowledgeStrip(
            content="""Procédure de création de compte utilisateur :
            1. Vérifier l'approbation du manager
            2. Créer le compte dans Active Directory
            3. Appliquer les groupes selon le rôle
            4. Envoyer les identifiants par canal sécurisé
            Note : Cette procédure date de 2021 et nécessite une mise à jour.""",
            strip_type="procedure_step",
            position=1,
            
            source_doc_id=2,
            domain_category="procedure",
            compliance_refs=[],
            context={
                'doc_type': 'procedure',
                'last_updated': '2021-06-10',
                'is_official': True,
                'version': '1.5'
            }
        ),

        # Strip 3 : FAQ informelle mais récente
        KnowledgeStrip(
            content="""Q: Comment demander un accès à un nouveau système ?
            R: Il faut remplir le formulaire de demande d'accès dans ServiceNow, qui sera 
            automatiquement routé vers votre manager pour approbation. Ensuite, l'équipe IT 
            traitera la demande sous 48h. Cette procédure a été mise à jour en mars 2024 
            pour inclure l'authentification MFA obligatoire.""",
            strip_type="faq_entry",
            position=2,
            
            source_doc_id=3,
            domain_category="faq",
            compliance_refs=[],
            context={
                'doc_type': 'faq',
                'last_updated': '2024-03-20',
                'is_official': False,
                'has_keywords': True
            }
        ),

        # Strip 4 : Rapport d'audit avec recommandations
        KnowledgeStrip(
            content="""Audit ISO 27001 - Constatation : Le processus de revue des accès 
            utilisateurs présente des lacunes. Recommandation : Implémenter une revue 
            trimestrielle automatisée des droits d'accès avec validation par les propriétaires 
            de données. Cette mesure répondrait pleinement à l'exigence A.9.2.5.""",
            strip_type="audit_finding",
            position=3,
            
            source_doc_id=4,
            domain_category="audit",
            compliance_refs=["ISO 27001 A.9.2.5"],
            context={
                'doc_type': 'audit_report',
                'last_updated': '2024-02-28',
                'is_official': True,
                'criticality': 'high'
            }
        ),

        # Strip 5 : Document technique sur les pare-feu (non pertinent)
        KnowledgeStrip(
            content="""Configuration avancée du pare-feu Palo Alto : Les règles de filtrage 
            doivent être configurées en mode strict avec inspection SSL/TLS activée. 
            Utiliser les profils de sécurité pour bloquer les menaces connues.""",
            strip_type="technical_guide",
            position=4,
            
            source_doc_id=5,
            domain_category="technical",
            compliance_refs=[],
            context={
                'doc_type': 'configuration_guide',
                'last_updated': '2024-01-10',
                'is_official': False
            }
        ),

        # Strip 6 : Matrice de contrôle
        KnowledgeStrip(
            content="""Contrôle A.9.1.1 - Politique de contrôle d'accès
            Statut : Conforme
            Description : Une politique de contrôle d'accès doit être établie, documentée et revue
            Mise en œuvre : Voir POL-SEC-001 Politique de gestion des accès
            Preuves : Politique signée, PV de revue annuelle, formations dispensées""",
            strip_type="control_measure",
            position=5,
            
            source_doc_id=6,
            domain_category="control",
            compliance_refs=["ISO 27001 A.9.1.1"],
            context={
                'doc_type': 'control_matrix',
                'last_updated': '2024-03-01',
                'is_official': True,
                'control_status': 'compliant'
            }
        )
    ]

    return strips


def test_audit_context_detection():
    """
    Teste la détection automatique du contexte d'audit.
    """
    print("=" * 80)
    print("TEST 1 : Détection du contexte d'audit")
    print("=" * 80)

    from src.grading.audit_relevance_rules import AuditRelevanceAnalyzer

    analyzer = AuditRelevanceAnalyzer()

    test_queries = [
        "Comment gérons-nous les accès conformément à ISO 27001 A.9 ?",
        "Quelle est notre procédure de notification en cas de violation de données personnelles selon le RGPD ?",
        "Quels sont les paramètres de hardening appliqués sur nos serveurs Linux ?",
        "Pouvez-vous décrire votre expérience dans l'implémentation de SIEM pour des clients similaires ?",
        "Quel est le processus d'escalade en cas d'incident de sécurité majeur ?"
    ]

    for query in test_queries:
        context = analyzer.detect_audit_context(query)
        print(f"\nQuestion : {query}")
        print(f"Contexte détecté : {context.value}")
        print("-" * 50)


def test_relevance_with_audit_rules():
    """
    Teste le grading avec et sans les règles d'audit.
    """
    print("\n" + "=" * 80)
    print("TEST 2 : Comparaison du grading avec/sans règles d'audit")
    print("=" * 80)

    # Créer les strips de test
    strips = create_test_strips()

    # Question d'audit ISO 27001
    query = "Quelle est notre politique de gestion des accès selon ISO 27001 A.9 ?"

    # Test SANS règles d'audit
    print("\n### SANS règles d'audit ###")
    grader_without = RelevanceGrader(use_llm=False, enable_audit_rules=False)
    results_without = grader_without.grade_strips(query, strips)

    print("\nRésultats (sans règles d'audit) :")
    for i, result in enumerate(results_without[:4]):
        print(f"{i + 1}. Score: {result.relevance_score:.3f} | "
              f"Type: {result.strip.context.get('doc_type')} | "
              f"Contenu: {result.strip.content[:60]}...")

    # Test AVEC règles d'audit
    print("\n### AVEC règles d'audit ###")
    grader_with = RelevanceGrader(use_llm=False, enable_audit_rules=True)

    # Ajouter des métadonnées d'audit
    audit_metadata = {
        'audit_type': 'ISO 27001',
        'audit_scope': 'Contrôle d\'accès',
        'audit_year': '2024'
    }

    results_with = grader_with.grade_strips(query, strips, audit_metadata=audit_metadata)

    print("\nRésultats (avec règles d'audit) :")
    for i, result in enumerate(results_with[:4]):
        print(f"{i + 1}. Score: {result.relevance_score:.3f} | "
              f"Type: {result.strip.context.get('doc_type')} | "
              f"Audit: {result.strip.context.get('audit_reasoning', 'N/A')}")
        print(f"   Contenu: {result.strip.content[:80]}...")

    # Analyser les différences
    print("\n### ANALYSE DES DIFFÉRENCES ###")
    for i in range(min(len(results_without), len(results_with))):
        strip_without = results_without[i]
        strip_with = results_with[i]

        if strip_without.strip.position == strip_with.strip.position:
            diff = strip_with.relevance_score - strip_without.relevance_score
            if abs(diff) > 0.05:
                print(f"\nStrip {strip_without.strip.position} - "
                      f"Type: {strip_without.strip.context.get('doc_type')}")
                print(f"  Score sans règles: {strip_without.relevance_score:.3f}")
                print(f"  Score avec règles: {strip_with.relevance_score:.3f}")
                print(f"  Différence: {diff:+.3f}")


def test_completeness_check():
    """
    Teste la vérification de complétude pour une réponse d'audit.
    """
    print("\n" + "=" * 80)
    print("TEST 3 : Vérification de la complétude de la réponse")
    print("=" * 80)

    strips = create_test_strips()
    grader = RelevanceGrader(use_llm=False, enable_audit_rules=True)

    # Question qui nécessite plusieurs éléments
    query = "Décrivez notre processus complet de gestion des accès ISO 27001, incluant politique, procédures et contrôles"

    results = grader.grade_strips(query, strips)

    # Vérifier la complétude
    if results and 'audit_completeness' in results[0].strip.context:
        completeness = results[0].strip.context['audit_completeness']
        print(f"\nComplétude de la réponse : {'✓ Complète' if completeness['is_complete'] else '✗ Incomplète'}")

        if not completeness['is_complete']:
            print("\nÉléments manquants identifiés :")
            for suggestion in completeness['missing_elements']:
                print(f"  - {suggestion}")

    # Afficher ce qui a été trouvé
    print("\nÉléments trouvés :")
    for result in results[:4]:
        if result.relevance_score > 0.5:
            print(f"  ✓ {result.strip.context.get('doc_type')} - "
                  f"{result.strip.context.get('last_updated', 'Date inconnue')}")


def test_multi_framework_mapping():
    """
    Teste la reconnaissance des mappings entre frameworks.
    """
    print("\n" + "=" * 80)
    print("TEST 4 : Mapping multi-frameworks")
    print("=" * 80)

    from src.grading.knowledge_stripper import KnowledgeStrip

    # Strip qui référence plusieurs frameworks
    multi_framework_strip = KnowledgeStrip(
        content="""Ce contrôle d'accès répond aux exigences suivantes :
        - ISO 27001 A.9.1 : Politique de contrôle d'accès
        - NIST AC-1 : Access Control Policy and Procedures
        - CIS Control 6 : Access Control Management
        Tous ces contrôles visent à établir une gouvernance des accès.""",
        strip_type="control_mapping",
        position=0,
        
        source_doc_id=10,
        domain_category="compliance",
        compliance_refs=["ISO 27001 A.9.1", "NIST AC-1", "CIS 6"],
        context={'doc_type': 'control_matrix'}
    )

    grader = RelevanceGrader(use_llm=False, enable_audit_rules=True)

    # Tester avec différentes questions référençant différents frameworks
    queries = [
        "Comment satisfaire ISO 27001 A.9 ?",
        "Quelle est notre conformité NIST AC-1 ?",
        "Avons-nous implémenté CIS Control 6 ?"
    ]

    for query in queries:
        results = grader.grade_strips(query, [multi_framework_strip])
        print(f"\nQuestion : {query}")
        print(f"Score : {results[0].relevance_score:.3f}")
        print(f"Raisonnement : {results[0].strip.context.get('audit_reasoning', 'N/A')}")


def main():
    """
    Lance tous les tests.
    """
    print("🔍 TESTS DU SYSTÈME DE GRADING AMÉLIORÉ POUR LES AUDITS\n")

    test_audit_context_detection()
    test_relevance_with_audit_rules()
    test_completeness_check()
    test_multi_framework_mapping()

    print("\n✅ Tests terminés !")
    print("\nPoints clés démontrés :")
    print("1. Détection automatique du contexte d'audit (ISO, RGPD, technique...)")
    print("2. Pondération selon l'autorité de la source (politique > FAQ)")
    print("3. Prise en compte de la fraîcheur des documents")
    print("4. Vérification de la complétude des réponses")
    print("5. Reconnaissance des équivalences entre frameworks")


if __name__ == "__main__":
    main()
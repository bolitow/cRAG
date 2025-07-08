# src/grading/audit_relevance_rules.py
"""
Règles de pertinence spécifiques aux audits de cybersécurité et appels d'offres.

Ce module contient toute l'intelligence métier pour évaluer la pertinence
des knowledge strips dans le contexte spécifique des audits. Il comprend
les nuances entre une question de conformité ISO 27001 et une exigence RGPD.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re


class AuditContext(Enum):
    """
    Types de contextes d'audit reconnus.

    Chaque contexte a ses propres critères de pertinence.
    """
    ISO_27001 = "iso_27001"
    ISO_27002 = "iso_27002"
    RGPD_GDPR = "rgpd_gdpr"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    TECHNICAL_AUDIT = "technical_audit"
    COMPLIANCE_GENERAL = "compliance_general"
    RFP_RESPONSE = "rfp_response"  # Réponse à appel d'offres
    INCIDENT_RESPONSE = "incident_response"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class AuditRelevanceRule:
    """
    Une règle de pertinence pour un contexte d'audit spécifique.

    Cette structure définit ce qui rend un strip pertinent pour
    un type particulier de question d'audit.
    """
    context: AuditContext
    required_elements: List[str]  # Éléments obligatoires
    preferred_elements: List[str]  # Éléments qui augmentent le score
    source_weights: Dict[str, float]  # Poids selon le type de document
    temporal_relevance: bool  # Si la fraîcheur du document importe
    authority_required: bool  # Si on a besoin d'une source officielle


class AuditRelevanceAnalyzer:
    """
    Analyseur spécialisé pour la pertinence dans le contexte des audits.

    Cette classe est comme un auditeur expert qui sait exactement
    quels documents chercher selon le type de question posée.
    """

    def __init__(self):
        # Initialiser les règles pour chaque contexte d'audit
        self._init_audit_rules()
        # Initialiser les patterns de détection de contexte
        self._init_context_patterns()
        # Initialiser les mappings de conformité
        self._init_compliance_mappings()

    def _init_audit_rules(self):
        """
        Définit les règles de pertinence pour chaque type d'audit.

        Ces règles reflètent l'expérience d'auditeurs chevronnés.
        """
        self.audit_rules = {
            AuditContext.ISO_27001: AuditRelevanceRule(
                context=AuditContext.ISO_27001,
                required_elements=[
                    "politique", "procédure", "contrôle", "mesure",
                    "responsabilité", "revue", "amélioration continue"
                ],
                preferred_elements=[
                    "pdca", "plan-do-check-act", "indicateur", "kpi",
                    "audit interne", "revue de direction", "non-conformité"
                ],
                source_weights={
                    "policy": 1.0,
                    "procedure": 0.9,
                    "standard": 0.8,
                    "audit_report": 0.7,
                    "control_matrix": 0.9,
                    "guideline": 0.5,
                    "faq": 0.3
                },
                temporal_relevance=True,  # Les politiques doivent être à jour
                authority_required=True  # Besoin de documents officiels
            ),

            AuditContext.RGPD_GDPR: AuditRelevanceRule(
                context=AuditContext.RGPD_GDPR,
                required_elements=[
                    "données personnelles", "personal data", "consentement",
                    "droits", "traitement", "responsable", "sous-traitant",
                    "dpo", "protection", "privacy"
                ],
                preferred_elements=[
                    "minimisation", "portabilité", "effacement", "rectification",
                    "notification", "violation", "impact assessment", "aipd",
                    "base légale", "finalité"
                ],
                source_weights={
                    "policy": 1.0,
                    "procedure": 0.9,
                    "privacy_notice": 1.0,
                    "dpia": 0.95,  # Data Protection Impact Assessment
                    "register": 0.9,  # Registre des traitements
                    "guideline": 0.6,
                    "faq": 0.4
                },
                temporal_relevance=True,
                authority_required=True
            ),

            AuditContext.TECHNICAL_AUDIT: AuditRelevanceRule(
                context=AuditContext.TECHNICAL_AUDIT,
                required_elements=[
                    "configuration", "paramètre", "architecture", "sécurité",
                    "vulnérabilité", "patch", "mise à jour", "firewall",
                    "chiffrement", "authentification"
                ],
                preferred_elements=[
                    "hardening", "baseline", "cis", "benchmark", "scan",
                    "pentest", "logs", "monitoring", "siem", "ids", "ips"
                ],
                source_weights={
                    "technical_standard": 1.0,
                    "configuration_guide": 0.95,
                    "architecture_document": 0.9,
                    "procedure": 0.8,
                    "audit_report": 0.85,
                    "incident_report": 0.7,
                    "faq": 0.3
                },
                temporal_relevance=True,  # Les configs changent souvent
                authority_required=False  # Les guides techniques suffisent
            ),

            AuditContext.RFP_RESPONSE: AuditRelevanceRule(
                context=AuditContext.RFP_RESPONSE,
                required_elements=[
                    "capacité", "expérience", "référence", "certification",
                    "méthodologie", "équipe", "compétence", "sla"
                ],
                preferred_elements=[
                    "cas client", "success story", "retour d'expérience",
                    "innovation", "valeur ajoutée", "différenciateur",
                    "roadmap", "evolution"
                ],
                source_weights={
                    "company_presentation": 0.9,
                    "case_study": 1.0,
                    "certification": 0.95,
                    "policy": 0.7,
                    "methodology": 0.9,
                    "faq": 0.5
                },
                temporal_relevance=True,  # Veulent du récent
                authority_required=False  # Marketing accepté
            ),

            AuditContext.INCIDENT_RESPONSE: AuditRelevanceRule(
                context=AuditContext.INCIDENT_RESPONSE,
                required_elements=[
                    "incident", "crise", "escalade", "notification",
                    "containment", "éradication", "recovery", "leçons"
                ],
                preferred_elements=[
                    "rto", "rpo", "playbook", "runbook", "contact",
                    "communication", "forensics", "timeline", "post-mortem"
                ],
                source_weights={
                    "incident_procedure": 1.0,
                    "crisis_plan": 0.95,
                    "contact_list": 0.9,
                    "incident_report": 0.85,
                    "lessons_learned": 0.8,
                    "policy": 0.6
                },
                temporal_relevance=True,  # Procédures actuelles critiques
                authority_required=True  # Besoin du process officiel
            )
        }

    def _init_context_patterns(self):
        """
        Patterns pour détecter automatiquement le contexte d'audit.

        Ces patterns permettent d'identifier le type d'audit à partir
        de la question posée, pour appliquer les bonnes règles.
        """
        self.context_patterns = {
            AuditContext.ISO_27001: [
                r'iso\s*27001', r'isms', r'smsi',
                r'système de management', r'management system',
                r'annexe a', r'annex a', r'contrôle\s+a\.\d+',
                r'statement of applicability', r'soa'
            ],

            AuditContext.RGPD_GDPR: [
                r'rgpd', r'gdpr', r'article\s+\d+\s+rgpd',
                r'données personnelles', r'personal data',
                r'privacy', r'vie privée', r'consentement',
                r'dpo', r'data protection officer'
            ],

            AuditContext.TECHNICAL_AUDIT: [
                r'configuration', r'technique', r'technical',
                r'architecture', r'infrastructure', r'réseau',
                r'firewall', r'serveur', r'vulnerability',
                r'hardening', r'patch', r'cve-\d{4}-\d+'
            ],

            AuditContext.RFP_RESPONSE: [
                r'appel d\'offre', r'rfp', r'rfq', r'tender',
                r'proposition', r'capacité', r'référence client',
                r'expérience', r'expertise', r'différenciateur'
            ],

            AuditContext.INCIDENT_RESPONSE: [
                r'incident', r'crise', r'crisis', r'breach',
                r'urgence', r'emergency', r'escalade',
                r'notification', r'recovery', r'rto', r'rpo'
            ]
        }

    def _init_compliance_mappings(self):
        """
        Mappings entre les différents frameworks de conformité.

        Permet de comprendre qu'une exigence ISO 27001 peut être
        satisfaite par un contrôle équivalent dans un autre framework.
        """
        self.compliance_mappings = {
            # ISO 27001 Annexe A vers autres frameworks
            "A.5": {  # Politiques de sécurité
                "nist": ["AC-1", "AT-1", "AU-1"],
                "cis": ["CIS 1", "CIS 2"],
                "description": "Politiques de sécurité de l'information"
            },
            "A.9": {  # Contrôle d'accès
                "nist": ["AC-2", "AC-3", "AC-6"],
                "cis": ["CIS 5", "CIS 6"],
                "rgpd": ["Article 32"],
                "description": "Gestion des accès et des identités"
            },
            "A.12.6": {  # Gestion des vulnérabilités
                "nist": ["RA-5", "SI-2"],
                "cis": ["CIS 3", "CIS 4"],
                "description": "Gestion des vulnérabilités techniques"
            },
            "A.16": {  # Gestion des incidents
                "nist": ["IR-4", "IR-5", "IR-6"],
                "rgpd": ["Article 33", "Article 34"],
                "description": "Gestion des incidents de sécurité"
            }
        }

    def detect_audit_context(self, query: str, metadata: Optional[Dict] = None) -> AuditContext:
        """
        Détecte automatiquement le contexte d'audit à partir de la question.

        Cette méthode analyse la question pour comprendre dans quel
        cadre d'audit nous nous trouvons.
        """
        query_lower = query.lower()

        # Vérifier les patterns de chaque contexte
        context_scores = {}

        for context, patterns in self.context_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1

            if score > 0:
                context_scores[context] = score

        # Si on a des métadonnées, les utiliser pour affiner
        if metadata and 'audit_type' in metadata:
            audit_type = metadata['audit_type'].lower()
            for context in AuditContext:
                if context.value in audit_type:
                    context_scores[context] = context_scores.get(context, 0) + 3

        # Retourner le contexte avec le score le plus élevé
        if context_scores:
            return max(context_scores, key=context_scores.get)

        # Par défaut, contexte de conformité générale
        return AuditContext.COMPLIANCE_GENERAL

    def calculate_audit_relevance_bonus(self,
                                        query: str,
                                        strip: 'KnowledgeStrip',
                                        base_score: float,
                                        metadata: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calcule un bonus de pertinence spécifique au contexte d'audit.

        Cette méthode ajuste le score de base en fonction des règles
        spécifiques au type d'audit détecté.

        Returns:
            Tuple (score_ajusté, explication)
        """
        # Détecter le contexte d'audit
        context = self.detect_audit_context(query, metadata)

        # Si pas de règle spécifique, retourner le score de base
        if context not in self.audit_rules:
            return base_score, "Contexte d'audit non spécifique"

        rule = self.audit_rules[context]
        adjustments = []

        # 1. Ajustement selon le type de source
        doc_type = strip.context.get('doc_type', 'general')
        if doc_type in rule.source_weights:
            weight = rule.source_weights[doc_type]
            base_score *= weight
            adjustments.append(f"Source {doc_type}: x{weight:.1f}")

        # 2. Bonus pour éléments requis
        content_lower = strip.content.lower()
        required_found = sum(1 for elem in rule.required_elements if elem in content_lower)
        if required_found > 0:
            bonus = min(0.2, required_found * 0.05)
            base_score += bonus
            adjustments.append(f"Éléments requis ({required_found}): +{bonus:.2f}")

        # 3. Bonus pour éléments préférés
        preferred_found = sum(1 for elem in rule.preferred_elements if elem in content_lower)
        if preferred_found > 0:
            bonus = min(0.1, preferred_found * 0.02)
            base_score += bonus
            adjustments.append(f"Éléments préférés ({preferred_found}): +{bonus:.2f}")

        # 4. Pénalité pour document périmé si pertinence temporelle requise
        if rule.temporal_relevance and 'last_updated' in strip.context:
            # Logique pour vérifier la fraîcheur du document
            # (simplifiée ici, à adapter selon vos besoins)
            age_penalty = 0  # À calculer selon la date
            if age_penalty > 0:
                base_score -= age_penalty
                adjustments.append(f"Document ancien: -{age_penalty:.2f}")

        # 5. Bonus pour autorité si requise
        if rule.authority_required and strip.context.get('is_official', False):
            base_score += 0.1
            adjustments.append("Document officiel: +0.10")

        # 6. Bonus pour mapping de conformité
        if context in [AuditContext.ISO_27001, AuditContext.RGPD_GDPR]:
            compliance_bonus = self._check_compliance_mapping(query, strip.content)
            if compliance_bonus > 0:
                base_score += compliance_bonus
                adjustments.append(f"Mapping conformité: +{compliance_bonus:.2f}")

        # Limiter le score entre 0 et 1
        final_score = max(0.0, min(1.0, base_score))

        # Créer l'explication
        explanation = f"Contexte {context.value}: " + ", ".join(adjustments)

        return final_score, explanation

    def _check_compliance_mapping(self, query: str, content: str) -> float:
        """
        Vérifie si le contenu référence des contrôles équivalents
        dans différents frameworks.

        Par exemple, si on cherche ISO 27001 A.9 mais que le document
        parle de NIST AC-2, on sait que c'est pertinent.
        """
        bonus = 0.0

        # Chercher des références à des contrôles
        iso_pattern = r'A\.\d+(?:\.\d+)?'
        nist_pattern = r'[A-Z]{2}-\d+'
        cis_pattern = r'CIS\s+\d+'

        iso_matches = re.findall(iso_pattern, content)
        nist_matches = re.findall(nist_pattern, content)
        cis_matches = re.findall(cis_pattern, content)

        # Si on trouve des mappings connus, bonus
        for iso_control in iso_matches:
            if iso_control in self.compliance_mappings:
                mapping = self.compliance_mappings[iso_control]
                # Vérifier si on a des contrôles équivalents
                for nist in nist_matches:
                    if nist in mapping.get('nist', []):
                        bonus += 0.05
                for cis in cis_matches:
                    if cis in mapping.get('cis', []):
                        bonus += 0.05

        return min(bonus, 0.15)  # Plafonner le bonus

    def get_audit_specific_keywords(self, context: AuditContext) -> Dict[str, List[str]]:
        """
        Retourne les mots-clés spécifiques à surveiller pour un contexte d'audit.

        Ces mots-clés peuvent être utilisés par le Knowledge Stripper
        pour mieux découper les documents selon le contexte.
        """
        if context not in self.audit_rules:
            return {}

        rule = self.audit_rules[context]

        return {
            'critical': rule.required_elements,
            'important': rule.preferred_elements,
            'contextual': self._get_contextual_keywords(context)
        }

    def _get_contextual_keywords(self, context: AuditContext) -> List[str]:
        """
        Retourne des mots-clés contextuels supplémentaires selon le type d'audit.
        """
        contextual_keywords = {
            AuditContext.ISO_27001: [
                'amélioration continue', 'revue de direction', 'objectifs de sécurité',
                'parties intéressées', 'contexte organisationnel', 'leadership'
            ],
            AuditContext.RGPD_GDPR: [
                'transfert international', 'pays tiers', 'clauses contractuelles',
                'privacy by design', 'accountability', 'registre des activités'
            ],
            AuditContext.TECHNICAL_AUDIT: [
                'baseline', 'hardening guide', 'vulnerability scan', 'penetration test',
                'security headers', 'tls configuration', 'cipher suites'
            ],
            AuditContext.RFP_RESPONSE: [
                'sla', 'kpi', 'governance', 'escalation', 'support levels',
                'pricing model', 'implementation timeline', 'project methodology'
            ]
        }

        return contextual_keywords.get(context, [])

    def suggest_missing_elements(self,
                                 query: str,
                                 found_strips: List['GradedStrip'],
                                 context: Optional[AuditContext] = None) -> List[str]:
        """
        Suggère les éléments manquants pour répondre complètement à une question d'audit.

        Cette méthode analyse ce qui a été trouvé et identifie ce qui
        manque selon les standards du type d'audit.
        """
        if not context:
            context = self.detect_audit_context(query)

        if context not in self.audit_rules:
            return []

        rule = self.audit_rules[context]

        # Analyser ce qu'on a trouvé
        found_elements = set()
        for strip in found_strips:
            content_lower = strip.strip.content.lower()
            for element in rule.required_elements + rule.preferred_elements:
                if element in content_lower:
                    found_elements.add(element)

        # Identifier ce qui manque
        missing_required = [elem for elem in rule.required_elements if elem not in found_elements]
        missing_preferred = [elem for elem in rule.preferred_elements if elem not in found_elements]

        suggestions = []

        if missing_required:
            suggestions.append(f"Éléments requis manquants pour {context.value}: {', '.join(missing_required[:3])}")

        if missing_preferred and len(suggestions) < 3:
            suggestions.append(f"Éléments recommandés manquants: {', '.join(missing_preferred[:3])}")

        # Suggestions spécifiques au contexte
        context_specific_suggestions = self._get_context_specific_suggestions(context, query)
        suggestions.extend(context_specific_suggestions)

        return suggestions[:5]  # Limiter à 5 suggestions

    def _get_context_specific_suggestions(self, context: AuditContext, query: str) -> List[str]:
        """
        Suggestions spécifiques selon le contexte d'audit.
        """
        suggestions = []

        if context == AuditContext.ISO_27001:
            if "politique" in query.lower():
                suggestions.append("Vérifier la date de dernière revue de la politique")
                suggestions.append("Confirmer l'approbation par la direction")
            elif "contrôle" in query.lower():
                suggestions.append("Vérifier la matrice d'applicabilité (SoA)")

        elif context == AuditContext.RGPD_GDPR:
            if "traitement" in query.lower():
                suggestions.append("Vérifier le registre des activités de traitement")
                suggestions.append("Confirmer la base légale du traitement")
            elif "droit" in query.lower():
                suggestions.append("Vérifier les procédures d'exercice des droits")

        elif context == AuditContext.TECHNICAL_AUDIT:
            suggestions.append("Vérifier les derniers rapports de scan de vulnérabilités")
            suggestions.append("Confirmer l'application des derniers patches de sécurité")

        return suggestions
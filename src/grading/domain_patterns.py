# src/grading/domain_patterns.py
"""
Patterns et vocabulaire spécifiques au domaine de la cybersécurité.

Ce module centralise toute la connaissance métier nécessaire pour identifier
et catégoriser correctement les éléments d'information dans vos documents
de politique de sécurité et de conformité.
"""


class CyberSecurityPatterns:
    """
    Cette classe contient tous les patterns linguistiques spécifiques
    à la cybersécurité et aux audits informatiques.
    """

    def __init__(self):
        # Initialiser tous les patterns de détection
        self._init_policy_patterns()
        self._init_control_patterns()
        self._init_compliance_patterns()
        self._init_procedure_patterns()
        self._init_risk_patterns()
        self._init_audit_patterns()

    def _init_policy_patterns(self):
        """
        Patterns pour identifier les déclarations de politique.
        Ces phrases établissent des règles ou des principes directeurs.
        """
        self.policy_indicators = {
            "mandatory": [
                "doit", "doivent", "est obligatoire", "sont obligatoires",
                "il est impératif", "il est requis", "exige", "impose",
                "shall", "must", "mandatory", "required"
            ],
            "forbidden": [
                "ne doit pas", "ne doivent pas", "est interdit", "sont interdits",
                "il est interdit", "prohibition", "shall not", "must not"
            ],
            "recommended": [
                "devrait", "devraient", "il est recommandé", "il est conseillé",
                "should", "recommended", "advised"
            ],
            "policy_declaration": [
                "la politique", "cette politique", "notre politique",
                "la présente politique", "politique de sécurité",
                "politique de", "policy states", "policy requires"
            ]
        }

        # Patterns regex pour détecter les structures de politique
        self.policy_patterns = [
            r"(?:La |Cette |Notre )?politique.*(?:stipule|établit|définit|exige)",
            r"Conformément à la politique",
            r"En vertu de.*politique",
            r"Article \d+.*:",  # Articles numérotés
            r"Section \d+.*:",  # Sections numérotées
            r"§\s*\d+",  # Paragraphes légaux
        ]

    def _init_control_patterns(self):
        """
        Patterns pour identifier les mesures de contrôle de sécurité.
        """
        self.control_keywords = {
            "technical_controls": [
                "pare-feu", "firewall", "antivirus", "chiffrement", "encryption",
                "authentification", "authentication", "mot de passe", "password",
                "certificat", "certificate", "vpn", "ids", "ips", "siem",
                "backup", "sauvegarde", "restauration", "recovery"
            ],
            "administrative_controls": [
                "procédure", "procedure", "processus", "process", "formation",
                "training", "sensibilisation", "awareness", "audit", "revue",
                "review", "approbation", "approval", "autorisation"
            ],
            "physical_controls": [
                "accès physique", "physical access", "badge", "biométrie",
                "biometric", "surveillance", "camera", "alarme", "alarm",
                "salle serveur", "data center", "coffre-fort", "safe"
            ]
        }

        self.control_patterns = [
            r"contrôle.*(?:technique|administratif|physique)",
            r"mesure de sécurité",
            r"mécanisme de protection",
            r"dispositif de sécurité",
            r"control.*implemented",
            r"security measure"
        ]

    def _init_compliance_patterns(self):
        """
        Patterns pour identifier les exigences de conformité.
        """
        self.compliance_frameworks = [
            "ISO 27001", "ISO 27002", "ISO 27701", "SOC 2", "SOC2",
            "RGPD", "GDPR", "PCI DSS", "PCI-DSS", "HIPAA", "SOX",
            "NIST", "ANSSI", "RGS", "eIDAS", "NIS", "LPM"
        ]

        self.compliance_keywords = [
            "conformité", "compliance", "exigence", "requirement",
            "norme", "standard", "réglementation", "regulation",
            "certification", "accréditation", "accreditation",
            "audit de conformité", "compliance audit"
        ]

        self.compliance_patterns = [
            r"(?:conformément|selon|en vertu de).*(?:" + "|".join(self.compliance_frameworks) + ")",
            r"exigence.*(?:légale|réglementaire|normative)",
            r"l'article.*(?:stipule|exige|impose)",
            r"requirement.*(?:states|mandates|requires)"
        ]

    def _init_procedure_patterns(self):
        """
        Patterns pour identifier les étapes de procédures.
        """
        self.procedure_indicators = {
            "sequence": [
                "étape", "step", "phase", "d'abord", "ensuite", "puis",
                "enfin", "finalement", "premièrement", "deuxièmement",
                "first", "then", "next", "finally"
            ],
            "action": [
                "effectuer", "réaliser", "vérifier", "valider", "approuver",
                "documenter", "notifier", "informer", "soumettre", "analyser",
                "perform", "execute", "verify", "validate", "approve"
            ],
            "responsibility": [
                "responsable", "responsible", "en charge", "doit", "must",
                "accountable", "owner", "RACI", "RASCI"
            ]
        }

        self.procedure_patterns = [
            r"\d+\.\s+",  # Numérotation 1. 2. 3.
            r"[a-z]\)\s+",  # Numérotation a) b) c)
            r"(?:Étape|Step)\s+\d+\s*:",
            r"(?:Phase|Stage)\s+\d+\s*:",
            r"→\s+",  # Flèches pour les étapes
            r"•\s+\w+.*:.*(?:doit|must|shall)"  # Bullets avec responsabilités
        ]

    def _init_risk_patterns(self):
        """
        Patterns pour identifier les éléments liés aux risques.
        """
        self.risk_keywords = {
            "risk_level": [
                "critique", "critical", "élevé", "high", "moyen", "medium",
                "faible", "low", "négligeable", "negligible"
            ],
            "risk_type": [
                "risque", "risk", "menace", "threat", "vulnérabilité",
                "vulnerability", "impact", "probabilité", "probability",
                "exposition", "exposure"
            ],
            "mitigation": [
                "mitigation", "atténuation", "réduction", "traitement",
                "treatment", "remédiation", "remediation", "contre-mesure",
                "countermeasure"
            ]
        }

        self.risk_patterns = [
            r"risque.*(?:élevé|moyen|faible|critique)",
            r"niveau de risque",
            r"évaluation.*risque",
            r"risk.*(?:high|medium|low|critical)",
            r"threat.*level",
            r"impact.*(?:business|operational|financial)"
        ]

    def _init_audit_patterns(self):
        """
        Patterns spécifiques aux audits et appels d'offres.
        """
        self.audit_keywords = {
            "evidence": [
                "preuve", "evidence", "justificatif", "démonstration",
                "attestation", "certificat", "rapport", "report",
                "documentation", "trace", "log", "journal"
            ],
            "audit_process": [
                "audit", "contrôle", "vérification", "inspection",
                "évaluation", "assessment", "review", "test",
                "échantillonnage", "sampling"
            ],
            "response_elements": [
                "réponse à l'exigence", "requirement response",
                "critère", "criterion", "indicateur", "indicator",
                "métrique", "metric", "KPI", "KRI"
            ]
        }

        self.audit_patterns = [
            r"(?:Pour|Afin de) répondre à.*exigence",
            r"(?:La|Notre) réponse.*(?:est|consiste)",
            r"Nous.*(?:mettons en œuvre|appliquons|respectons)",
            r"Evidence.*(?:provided|available|documented)",
            r"Critère.*(?:satisfait|rempli|atteint)"
        ]

    def get_pattern_category(self, text: str) -> str:
        """
        Détermine la catégorie principale d'un texte basé sur les patterns détectés.

        Returns:
            La catégorie dominante ou 'general' si aucune n'est détectée
        """
        text_lower = text.lower()
        scores = {
            "policy": 0,
            "control": 0,
            "compliance": 0,
            "procedure": 0,
            "risk": 0,
            "audit": 0
        }

        # Compter les occurrences de chaque type de pattern
        for keyword in self.policy_indicators.get("mandatory", []):
            if keyword in text_lower:
                scores["policy"] += 2

        for keyword in self.control_keywords.get("technical_controls", []):
            if keyword in text_lower:
                scores["control"] += 1

        for framework in self.compliance_frameworks:
            if framework.lower() in text_lower:
                scores["compliance"] += 3

        for keyword in self.procedure_indicators.get("sequence", []):
            if keyword in text_lower:
                scores["procedure"] += 1

        for keyword in self.risk_keywords.get("risk_type", []):
            if keyword in text_lower:
                scores["risk"] += 1

        # Retourner la catégorie avec le score le plus élevé
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        return "general"

    def extract_compliance_references(self, text: str) -> list:
        """
        Extrait toutes les références à des normes ou réglementations.

        Returns:
            Liste des normes/réglementations mentionnées
        """
        references = []
        text_upper = text.upper()

        for framework in self.compliance_frameworks:
            if framework.upper() in text_upper:
                references.append(framework)

        return list(set(references))  # Éliminer les doublons

    def identify_control_types(self, text: str) -> list:
        """
        Identifie les types de contrôles mentionnés dans le texte.

        Returns:
            Liste des types de contrôles (technical, administrative, physical)
        """
        control_types = []
        text_lower = text.lower()

        for control_type, keywords in self.control_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    control_types.append(control_type)
                    break

        return list(set(control_types))
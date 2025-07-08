# src/ingestion/preprocessors/security_preprocessor.py
import re
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..parsers.base_parser import ParsedDocument


@dataclass
class SecurityContext:
    """
    Contexte de sécurité extrait d'un document.

    Cette structure capture toutes les informations pertinentes
    pour la cybersécurité extraites lors du preprocessing.
    """
    compliance_frameworks: List[str]  # Normes référencées (ISO 27001, etc.)
    security_domains: List[str]  # Domaines (réseau, accès, crypto, etc.)
    risk_levels: List[str]  # Niveaux de risque mentionnés
    control_types: List[str]  # Types de contrôles (technique, admin, etc.)
    technologies: List[str]  # Technologies mentionnées
    threat_references: List[str]  # Menaces ou vulnérabilités référencées
    regulatory_requirements: List[str]  # Exigences réglementaires
    security_roles: List[str]  # Rôles (RSSI, DPO, etc.)
    time_constraints: List[Dict]  # Délais et échéances
    criticality_indicators: Dict[str, int]  # Indicateurs de criticité


class SecurityPreprocessor:
    """
    Préprocesseur spécialisé pour les documents de cybersécurité.

    Ce préprocesseur enrichit les documents parsés en :
    - Extrayant les références normatives et réglementaires
    - Identifiant les domaines de sécurité couverts
    - Détectant les niveaux de criticité et d'urgence
    - Normalisant la terminologie technique
    - Extrayant les contraintes temporelles
    - Créant des liens entre concepts liés

    C'est comme un expert en cybersécurité qui lit le document
    et ajoute des post-its avec des informations contextuelles importantes.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le préprocesseur avec sa configuration.

        Args:
            config: Configuration incluant :
                - extract_dates: Extraire les dates et délais
                - normalize_terms: Normaliser la terminologie
                - enrich_metadata: Enrichir les métadonnées
                - detect_references: Détecter les références croisées
                - language: Langue principale (fr/en)
        """
        self.config = config or {}

        # Configuration
        self.extract_dates = self.config.get('extract_dates', True)
        self.normalize_terms = self.config.get('normalize_terms', True)
        self.enrich_metadata = self.config.get('enrich_metadata', True)
        self.detect_references = self.config.get('detect_references', True)
        self.language = self.config.get('language', 'fr')

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialiser les bases de connaissances
        self._init_knowledge_bases()

        # Compiler les patterns regex
        self._compile_patterns()

    def _init_knowledge_bases(self):
        """
        Initialise les bases de connaissances du domaine cybersécurité.

        Ces dictionnaires constituent la connaissance métier du préprocesseur.
        """
        # Frameworks de conformité reconnus
        self.compliance_frameworks = {
            # ISO
            'ISO 27001': ['ISO27001', 'ISO 27001', 'ISO/IEC 27001', 'ISO/CEI 27001'],
            'ISO 27002': ['ISO27002', 'ISO 27002', 'ISO/IEC 27002'],
            'ISO 27005': ['ISO27005', 'ISO 27005', 'ISO/IEC 27005'],
            'ISO 27017': ['ISO27017', 'ISO 27017', 'ISO/IEC 27017'],
            'ISO 27018': ['ISO27018', 'ISO 27018', 'ISO/IEC 27018'],
            'ISO 27701': ['ISO27701', 'ISO 27701', 'ISO/IEC 27701'],
            'ISO 22301': ['ISO22301', 'ISO 22301', 'ISO/IEC 22301'],
            # Réglementations
            'RGPD': ['RGPD', 'GDPR', 'Règlement Général sur la Protection des Données'],
            'NIS': ['NIS', 'Directive NIS', 'Network and Information Security'],
            'NIS2': ['NIS2', 'NIS 2', 'Directive NIS 2'],
            'LPM': ['LPM', 'Loi de Programmation Militaire'],
            'eIDAS': ['eIDAS', 'electronic IDentification Authentication and trust Services'],
            # Standards sectoriels
            'PCI DSS': ['PCI DSS', 'PCI-DSS', 'Payment Card Industry Data Security Standard'],
            'HIPAA': ['HIPAA', 'Health Insurance Portability and Accountability Act'],
            'SOX': ['SOX', 'Sarbanes-Oxley', 'Sarbanes Oxley'],
            'SOC 2': ['SOC 2', 'SOC2', 'Service Organization Control 2'],
            # Frameworks
            'NIST CSF': ['NIST CSF', 'NIST Cybersecurity Framework', 'NIST Framework'],
            'CIS Controls': ['CIS Controls', 'CIS', 'Center for Internet Security'],
            'ANSSI': ['ANSSI', 'Agence Nationale de la Sécurité des Systèmes d\'Information'],
            'RGS': ['RGS', 'Référentiel Général de Sécurité'],
            'EBIOS': ['EBIOS', 'EBIOS Risk Manager', 'EBIOS RM']
        }

        # Domaines de sécurité
        self.security_domains = {
            'Gestion des accès': [
                'gestion des accès', 'access management', 'IAM',
                'authentification', 'authentication', 'autorisation',
                'authorization', 'privilèges', 'privileges', 'RBAC',
                'contrôle d\'accès', 'access control', 'SSO', 'MFA', '2FA'
            ],
            'Sécurité réseau': [
                'sécurité réseau', 'network security', 'pare-feu',
                'firewall', 'IDS', 'IPS', 'DMZ', 'segmentation',
                'VPN', 'NAC', 'proxy', 'WAF'
            ],
            'Cryptographie': [
                'cryptographie', 'cryptography', 'chiffrement',
                'encryption', 'PKI', 'certificat', 'certificate',
                'TLS', 'SSL', 'AES', 'RSA', 'hash', 'signature'
            ],
            'Sécurité des données': [
                'sécurité des données', 'data security', 'classification',
                'DLP', 'Data Loss Prevention', 'pseudonymisation',
                'anonymisation', 'minimisation', 'retention'
            ],
            'Continuité': [
                'continuité', 'continuity', 'PCA', 'PRA', 'DRP',
                'Business Continuity', 'Disaster Recovery', 'RTO', 'RPO',
                'sauvegarde', 'backup', 'restauration', 'recovery'
            ],
            'Gestion des incidents': [
                'incident', 'SIEM', 'SOC', 'CERT', 'CSIRT',
                'forensics', 'investigation', 'réponse', 'response',
                'escalade', 'escalation', 'crise', 'crisis'
            ],
            'Gouvernance': [
                'gouvernance', 'governance', 'politique', 'policy',
                'procédure', 'procedure', 'RACI', 'comité', 'committee',
                'risque', 'risk', 'conformité', 'compliance'
            ],
            'Sécurité physique': [
                'sécurité physique', 'physical security', 'badge',
                'biométrie', 'biometric', 'surveillance', 'CCTV',
                'contrôle d\'accès physique', 'datacenter'
            ]
        }

        # Niveaux de risque/criticité
        self.risk_levels = {
            'Critique': ['critique', 'critical', 'très élevé', 'very high', 'P1', 'sévérité 1'],
            'Élevé': ['élevé', 'high', 'important', 'majeur', 'major', 'P2', 'sévérité 2'],
            'Moyen': ['moyen', 'medium', 'modéré', 'moderate', 'P3', 'sévérité 3'],
            'Faible': ['faible', 'low', 'mineur', 'minor', 'P4', 'sévérité 4'],
            'Négligeable': ['négligeable', 'negligible', 'très faible', 'very low']
        }

        # Technologies et outils
        self.security_technologies = {
            'IAM': ['Active Directory', 'AD', 'LDAP', 'Okta', 'Auth0', 'Keycloak', 'SAML', 'OAuth', 'OIDC'],
            'SIEM': ['Splunk', 'QRadar', 'ArcSight', 'Elastic SIEM', 'Sentinel', 'LogRhythm'],
            'Firewall': ['Palo Alto', 'Fortinet', 'Cisco ASA', 'pfSense', 'Checkpoint', 'Sophos'],
            'Antivirus': ['CrowdStrike', 'SentinelOne', 'Symantec', 'McAfee', 'Kaspersky', 'ESET'],
            'Vulnerability': ['Qualys', 'Nessus', 'Rapid7', 'OpenVAS', 'Metasploit'],
            'Encryption': ['HSM', 'TPM', 'KMS', 'Vault', 'Let\'s Encrypt'],
            'Cloud': ['AWS', 'Azure', 'GCP', 'OCI', 'CloudFlare', 'Akamai']
        }

        # Rôles et responsabilités
        self.security_roles = {
            'RSSI': ['RSSI', 'CISO', 'Chief Information Security Officer',
                     'Responsable de la Sécurité des Systèmes d\'Information'],
            'DPO': ['DPO', 'Data Protection Officer', 'Délégué à la Protection des Données'],
            'DSI': ['DSI', 'CIO', 'Chief Information Officer', 'Directeur des Systèmes d\'Information'],
            'Administrateur': ['administrateur', 'admin', 'sysadmin', 'administrator'],
            'Auditeur': ['auditeur', 'auditor', 'auditeur interne', 'auditeur externe'],
            'SOC Analyst': ['analyste SOC', 'SOC analyst', 'opérateur SOC'],
            'Architect': ['architecte sécurité', 'security architect', 'architecte SI']
        }

        # Termes à normaliser
        self.term_normalizations = {
            # Français -> Terme normalisé
            'mot de passe': 'password',
            'pare-feu': 'firewall',
            'sauvegarde': 'backup',
            'chiffrement': 'encryption',
            'authentification': 'authentication',
            'autorisation': 'authorization',
            'vulnérabilité': 'vulnerability',
            'menace': 'threat',
            'risque': 'risk',
            'incident': 'incident',
            'conformité': 'compliance',
            'audit': 'audit',
            'contrôle': 'control',
            'accès': 'access',
            'identité': 'identity',
            'certificat': 'certificate',
            'clé': 'key',
            'réseau': 'network',
            'données': 'data',
            'système': 'system',
            'sécurité': 'security',
            'protection': 'protection',
            'politique': 'policy',
            'procédure': 'procedure',
            'processus': 'process'
        }

    def _compile_patterns(self):
        """
        Compile les patterns regex pour l'extraction d'informations.
        """
        # Pattern pour les dates et délais
        self.date_patterns = [
            # Dates explicites
            re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'),
            re.compile(
                r'\b(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{2,4})\b',
                re.IGNORECASE),
            re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(\d{2,4})\b',
                       re.IGNORECASE),
            # Délais relatifs
            re.compile(
                r'\b(\d+)\s*(jour|jours|day|days|semaine|semaines|week|weeks|mois|month|months|an|ans|year|years)\b',
                re.IGNORECASE),
            re.compile(r'\b(immédiat|immediate|urgent|dans les plus brefs délais|as soon as possible|asap)\b',
                       re.IGNORECASE),
            # Périodicité
            re.compile(
                r'\b(quotidien|daily|hebdomadaire|weekly|mensuel|monthly|trimestriel|quarterly|annuel|annual|yearly)\b',
                re.IGNORECASE)
        ]

        # Pattern pour les références de documents
        self.reference_patterns = [
            re.compile(
                r'\b(?:voir|see|cf\.?|référence|reference)\s+(?:document|section|article|chapitre|chapter)\s+([A-Z0-9\-\.]+)\b',
                re.IGNORECASE),
            re.compile(
                r'\b(?:conformément à|selon|as per|according to)\s+(?:la|le|l\')?(?:article|section|chapitre)?\s*([A-Z0-9\-\.]+)\b',
                re.IGNORECASE),
            re.compile(r'\[([A-Z0-9\-\.]+)\]'),  # Références entre crochets
            re.compile(r'\b(POL|PROC|STD|GUI)-\d{3,}\b')  # Codes de documents
        ]

        # Pattern pour les exigences
        self.requirement_patterns = [
            re.compile(r'\b(doit|doivent|must|shall|devra|devront)\b', re.IGNORECASE),
            re.compile(r'\b(obligatoire|mandatory|requis|required|impératif|imperative)\b', re.IGNORECASE),
            re.compile(r'\b(ne doit pas|ne doivent pas|must not|shall not|interdit|forbidden|prohibited)\b',
                       re.IGNORECASE)
        ]

        # Pattern pour les CVE et vulnérabilités
        self.vulnerability_patterns = [
            re.compile(r'\bCVE-\d{4}-\d{4,}\b'),
            re.compile(r'\bCVSS\s*:?\s*(\d+\.?\d*)\b', re.IGNORECASE),
            re.compile(r'\b(0-day|zero-day|0day)\b', re.IGNORECASE)
        ]

    def preprocess(self, parsed_doc: ParsedDocument) -> ParsedDocument:
        """
        Préprocesse un document parsé pour enrichir ses métadonnées.

        Cette méthode est le point d'entrée principal qui orchestre
        tout le preprocessing.

        Args:
            parsed_doc: Document parsé à enrichir

        Returns:
            Document enrichi avec les métadonnées de sécurité
        """
        self.logger.info(f"Preprocessing du document : {parsed_doc.metadata.get('file_name', 'Unknown')}")

        # Extraire le contexte de sécurité
        security_context = self._extract_security_context(parsed_doc)

        # Enrichir les métadonnées
        if self.enrich_metadata:
            self._enrich_metadata(parsed_doc, security_context)

        # Normaliser les termes si demandé
        if self.normalize_terms:
            self._normalize_terminology(parsed_doc)

        # Extraire les dates et contraintes temporelles
        if self.extract_dates:
            time_constraints = self._extract_time_constraints(parsed_doc)
            parsed_doc.metadata['time_constraints'] = time_constraints

        # Détecter les références croisées
        if self.detect_references:
            references = self._detect_cross_references(parsed_doc)
            parsed_doc.metadata['cross_references'] = references

        # Calculer des scores de pertinence
        relevance_scores = self._calculate_relevance_scores(parsed_doc, security_context)
        parsed_doc.metadata['relevance_scores'] = relevance_scores

        # Ajouter le contexte de sécurité aux métadonnées
        parsed_doc.metadata['security_context'] = {
            'compliance_frameworks': security_context.compliance_frameworks,
            'security_domains': security_context.security_domains,
            'risk_levels': security_context.risk_levels,
            'control_types': security_context.control_types,
            'technologies': security_context.technologies,
            'threat_references': security_context.threat_references,
            'regulatory_requirements': security_context.regulatory_requirements,
            'security_roles': security_context.security_roles,
            'criticality_indicators': security_context.criticality_indicators
        }

        # Marquer le document comme préprocessé
        parsed_doc.metadata['preprocessed'] = True
        parsed_doc.metadata['preprocessing_timestamp'] = datetime.now().isoformat()
        parsed_doc.metadata['preprocessor_version'] = '1.0'

        self.logger.info(
            f"Preprocessing terminé : "
            f"{len(security_context.compliance_frameworks)} frameworks, "
            f"{len(security_context.security_domains)} domaines détectés"
        )

        return parsed_doc

    def _extract_security_context(self, parsed_doc: ParsedDocument) -> SecurityContext:
        """
        Extrait le contexte de sécurité complet du document.
        """
        content = parsed_doc.content.lower()

        context = SecurityContext(
            compliance_frameworks=[],
            security_domains=[],
            risk_levels=[],
            control_types=[],
            technologies=[],
            threat_references=[],
            regulatory_requirements=[],
            security_roles=[],
            time_constraints=[],
            criticality_indicators={}
        )

        # Extraire les frameworks de conformité
        for framework, variations in self.compliance_frameworks.items():
            for variation in variations:
                if variation.lower() in content:
                    context.compliance_frameworks.append(framework)
                    break

        # Extraire les domaines de sécurité
        for domain, keywords in self.security_domains.items():
            domain_score = sum(1 for keyword in keywords if keyword.lower() in content)
            if domain_score > 0:
                context.security_domains.append(domain)
                # Plus il y a de mots-clés, plus le domaine est important
                context.criticality_indicators[f'domain_{domain}'] = domain_score

        # Extraire les niveaux de risque
        for level, variations in self.risk_levels.items():
            for variation in variations:
                if variation.lower() in content:
                    context.risk_levels.append(level)
                    break

        # Extraire les technologies
        for category, techs in self.security_technologies.items():
            for tech in techs:
                if tech.lower() in content:
                    context.technologies.append(tech)

        # Extraire les rôles
        for role, variations in self.security_roles.items():
            for variation in variations:
                if variation.lower() in content:
                    context.security_roles.append(role)
                    break

        # Extraire les vulnérabilités
        for match in self.vulnerability_patterns[0].finditer(parsed_doc.content):
            context.threat_references.append(match.group(0))

        # Détecter les types de contrôles
        control_keywords = {
            'Technique': ['technique', 'technical', 'logiciel', 'software', 'système'],
            'Administratif': ['administratif', 'administrative', 'procédure', 'processus'],
            'Physique': ['physique', 'physical', 'bâtiment', 'facility']
        }

        for control_type, keywords in control_keywords.items():
            if any(keyword in content for keyword in keywords):
                context.control_types.append(control_type)

        # Détecter les exigences réglementaires
        if any(pattern.search(parsed_doc.content) for pattern in self.requirement_patterns):
            context.regulatory_requirements.append('Contient des exigences obligatoires')

        # Calculer des indicateurs de criticité
        context.criticality_indicators['requirement_count'] = len(
            self.requirement_patterns[0].findall(parsed_doc.content)
        )
        context.criticality_indicators['vulnerability_count'] = len(context.threat_references)

        # Déduplication
        context.compliance_frameworks = list(set(context.compliance_frameworks))
        context.security_domains = list(set(context.security_domains))
        context.risk_levels = list(set(context.risk_levels))
        context.technologies = list(set(context.technologies))
        context.security_roles = list(set(context.security_roles))

        return context

    def _enrich_metadata(self, parsed_doc: ParsedDocument, security_context: SecurityContext):
        """
        Enrichit les métadonnées du document avec le contexte de sécurité.
        """
        # Déterminer le type de document de sécurité
        doc_content = parsed_doc.content.lower()

        security_doc_types = {
            'security_policy': ['politique de sécurité', 'security policy', 'pssi'],
            'security_procedure': ['procédure de sécurité', 'security procedure', 'mode opératoire'],
            'risk_assessment': ['analyse de risque', 'risk assessment', 'ebios'],
            'audit_report': ['rapport d\'audit', 'audit report', 'compte-rendu d\'audit'],
            'incident_report': ['rapport d\'incident', 'incident report', 'fiche d\'incident'],
            'compliance_matrix': ['matrice de conformité', 'compliance matrix', 'référentiel'],
            'security_standard': ['standard de sécurité', 'security standard', 'norme'],
            'security_guideline': ['guide de sécurité', 'security guideline', 'recommandations']
        }

        for doc_type, indicators in security_doc_types.items():
            if any(indicator in doc_content for indicator in indicators):
                parsed_doc.metadata['security_doc_type'] = doc_type
                break

        # Ajouter un score d'importance basé sur plusieurs facteurs
        importance_score = 0

        # Facteur 1 : Nombre de frameworks de conformité
        importance_score += len(security_context.compliance_frameworks) * 2

        # Facteur 2 : Présence d'exigences obligatoires
        if security_context.regulatory_requirements:
            importance_score += 3

        # Facteur 3 : Niveau de risque mentionné
        if 'Critique' in security_context.risk_levels:
            importance_score += 5
        elif 'Élevé' in security_context.risk_levels:
            importance_score += 3

        # Facteur 4 : Rôles critiques mentionnés
        if 'RSSI' in security_context.security_roles or 'DPO' in security_context.security_roles:
            importance_score += 2

        parsed_doc.metadata['importance_score'] = min(10, importance_score)  # Score sur 10

        # Ajouter des tags pour faciliter la recherche
        tags = []
        tags.extend([f"compliance:{fw}" for fw in security_context.compliance_frameworks])
        tags.extend([f"domain:{domain}" for domain in security_context.security_domains])
        tags.extend([f"tech:{tech}" for tech in security_context.technologies[:5]])  # Limiter

        parsed_doc.metadata['security_tags'] = tags

        # Indicateur de document critique
        is_critical = (
                importance_score >= 7 or
                'Critique' in security_context.risk_levels or
                len(security_context.threat_references) > 0
        )
        parsed_doc.metadata['is_critical_security_doc'] = is_critical

    def _normalize_terminology(self, parsed_doc: ParsedDocument):
        """
        Normalise la terminologie dans le document.

        Cette normalisation aide à la recherche en unifiant les termes
        équivalents (ex: "mot de passe" -> "password").
        """
        # Créer une version normalisée du contenu
        normalized_content = parsed_doc.content

        # Appliquer les normalisations
        for french_term, normalized_term in self.term_normalizations.items():
            # Créer un pattern qui préserve la casse et les limites de mots
            pattern = re.compile(r'\b' + re.escape(french_term) + r'\b', re.IGNORECASE)

            # Remplacer en préservant la casse originale
            def replace_preserve_case(match):
                original = match.group(0)
                if original.isupper():
                    return normalized_term.upper()
                elif original[0].isupper():
                    return normalized_term.capitalize()
                else:
                    return normalized_term

            normalized_content = pattern.sub(replace_preserve_case, normalized_content)

        # Stocker les deux versions
        parsed_doc.metadata['normalized_content'] = normalized_content
        parsed_doc.metadata['normalization_applied'] = True

    def _extract_time_constraints(self, parsed_doc: ParsedDocument) -> List[Dict]:
        """
        Extrait les contraintes temporelles du document.

        Les délais et échéances sont critiques pour la conformité.
        """
        time_constraints = []

        # Rechercher les dates et délais
        for pattern in self.date_patterns:
            for match in pattern.finditer(parsed_doc.content):
                constraint = {
                    'text': match.group(0),
                    'type': 'unknown',
                    'position': match.start()
                }

                # Déterminer le type de contrainte
                context = parsed_doc.content[max(0, match.start() - 50):match.end() + 50]

                if any(word in context.lower() for word in ['échéance', 'deadline', 'avant le', 'before']):
                    constraint['type'] = 'deadline'
                elif any(word in context.lower() for word in ['délai', 'dans les', 'within']):
                    constraint['type'] = 'delay'
                elif any(word in context.lower() for word in ['périod', 'chaque', 'every', 'tous les']):
                    constraint['type'] = 'periodic'
                elif any(word in context.lower() for word in ['immédiat', 'urgent', 'immediate']):
                    constraint['type'] = 'urgent'
                    constraint['priority'] = 'high'

                # Parser la date si possible
                constraint['parsed_date'] = self._parse_date_string(match.group(0))

                time_constraints.append(constraint)

        return time_constraints

    def _parse_date_string(self, date_str: str) -> Optional[str]:
        """
        Tente de parser une chaîne de date en format ISO.
        """
        # Mapping des mois français
        month_mapping = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }

        try:
            # Essayer différents formats
            # Format DD/MM/YYYY ou DD-MM-YYYY
            match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_str)
            if match:
                day, month, year = match.groups()
                year = int(year)
                if year < 100:
                    year += 2000
                return f"{year:04d}-{int(month):02d}-{int(day):02d}"

            # Format avec mois en texte
            for month_name, month_num in month_mapping.items():
                if month_name in date_str.lower():
                    match = re.search(r'(\d{1,2})\s+' + month_name + r'\s+(\d{2,4})', date_str.lower())
                    if match:
                        day, year = match.groups()
                        year = int(year)
                        if year < 100:
                            year += 2000
                        return f"{year:04d}-{month_num:02d}-{int(day):02d}"

            # Délais relatifs
            match = re.search(r'(\d+)\s*(jour|day|semaine|week|mois|month)', date_str.lower())
            if match:
                amount, unit = match.groups()
                amount = int(amount)

                # Calculer la date relative
                from datetime import datetime, timedelta
                base_date = datetime.now()

                if 'jour' in unit or 'day' in unit:
                    target_date = base_date + timedelta(days=amount)
                elif 'semaine' in unit or 'week' in unit:
                    target_date = base_date + timedelta(weeks=amount)
                elif 'mois' in unit or 'month' in unit:
                    target_date = base_date + timedelta(days=amount * 30)  # Approximation

                return target_date.strftime('%Y-%m-%d')

        except:
            pass

        return None

    def _detect_cross_references(self, parsed_doc: ParsedDocument) -> List[Dict]:
        """
        Détecte les références à d'autres documents ou sections.
        """
        references = []

        for pattern in self.reference_patterns:
            for match in pattern.finditer(parsed_doc.content):
                reference = {
                    'text': match.group(0),
                    'reference': match.group(1) if match.lastindex else match.group(0),
                    'position': match.start(),
                    'type': 'unknown'
                }

                # Déterminer le type de référence
                context = match.group(0).lower()
                if any(word in context for word in ['politique', 'policy', 'pol-']):
                    reference['type'] = 'policy'
                elif any(word in context for word in ['procédure', 'procedure', 'proc-']):
                    reference['type'] = 'procedure'
                elif any(word in context for word in ['standard', 'norme', 'std-']):
                    reference['type'] = 'standard'
                elif any(word in context for word in ['guide', 'guideline', 'gui-']):
                    reference['type'] = 'guideline'

                references.append(reference)

        return references

    def _calculate_relevance_scores(self,
                                    parsed_doc: ParsedDocument,
                                    security_context: SecurityContext) -> Dict[str, float]:
        """
        Calcule des scores de pertinence pour différents cas d'usage.
        """
        scores = {}

        # Score pour audit de conformité
        audit_score = 0.0
        if security_context.compliance_frameworks:
            audit_score += 0.3
        if security_context.regulatory_requirements:
            audit_score += 0.3
        if 'compliance' in parsed_doc.doc_type.lower() or 'audit' in parsed_doc.doc_type.lower():
            audit_score += 0.4
        scores['audit_relevance'] = min(1.0, audit_score)

        # Score pour gestion des risques
        risk_score = 0.0
        if security_context.risk_levels:
            risk_score += 0.3
        if security_context.threat_references:
            risk_score += 0.3
        if 'risk' in parsed_doc.doc_type.lower():
            risk_score += 0.4
        scores['risk_relevance'] = min(1.0, risk_score)

        # Score pour implémentation technique
        tech_score = 0.0
        if security_context.technologies:
            tech_score += 0.3
        if 'Technique' in security_context.control_types:
            tech_score += 0.3
        if any(domain in ['Sécurité réseau', 'Cryptographie', 'Gestion des accès']
               for domain in security_context.security_domains):
            tech_score += 0.4
        scores['technical_relevance'] = min(1.0, tech_score)

        # Score pour gouvernance
        gov_score = 0.0
        if security_context.security_roles:
            gov_score += 0.3
        if 'Gouvernance' in security_context.security_domains:
            gov_score += 0.4
        if 'policy' in parsed_doc.doc_type.lower():
            gov_score += 0.3
        scores['governance_relevance'] = min(1.0, gov_score)

        # Score global
        scores['overall_relevance'] = sum(scores.values()) / len(scores)

        return scores

    def batch_preprocess(self,
                         parsed_documents: List[ParsedDocument],
                         progress_callback: Optional[callable] = None) -> List[ParsedDocument]:
        """
        Préprocesse un lot de documents.

        Args:
            parsed_documents: Liste des documents à préprocesser
            progress_callback: Fonction appelée avec (current, total)

        Returns:
            Liste des documents préprocessés
        """
        preprocessed = []
        total = len(parsed_documents)

        for i, doc in enumerate(parsed_documents):
            try:
                preprocessed_doc = self.preprocess(doc)
                preprocessed.append(preprocessed_doc)

                if progress_callback:
                    progress_callback(i + 1, total)

            except Exception as e:
                self.logger.error(
                    f"Erreur lors du preprocessing de {doc.metadata.get('file_name', 'Unknown')}: {e}"
                )
                # Ajouter le document non préprocessé avec un flag d'erreur
                doc.metadata['preprocessing_error'] = str(e)
                preprocessed.append(doc)

        return preprocessed
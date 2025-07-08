# src/ingestion/preprocessors/metadata_extractor.py
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, date
from pathlib import Path
import logging
from dataclasses import dataclass, field

from ..parsers.base_parser import ParsedDocument


@dataclass
class DocumentMetadata:
    """
    Structure complète des métadonnées extraites d'un document.

    Cette classe capture toutes les informations contextuelles
    qui peuvent être utiles pour la recherche et la conformité.
    """
    # Identification
    document_id: str  # ID unique généré
    document_code: Optional[str]  # Code officiel (ex: POL-SEC-001)
    version: Optional[str]  # Version du document
    revision: Optional[str]  # Révision/amendement

    # Classification
    classification_level: str  # Public, Interne, Confidentiel, Secret
    data_sensitivity: List[str]  # Types de données sensibles
    handling_restrictions: List[str]  # Restrictions de manipulation

    # Cycle de vie
    creation_date: Optional[date]  # Date de création
    last_modified: Optional[date]  # Dernière modification
    review_date: Optional[date]  # Prochaine révision
    expiry_date: Optional[date]  # Date d'expiration
    lifecycle_stage: str  # Draft, Active, Under Review, Obsolete

    # Responsabilités
    author: Optional[str]  # Auteur principal
    contributors: List[str]  # Contributeurs
    owner: Optional[str]  # Propriétaire du document
    approver: Optional[str]  # Approbateur
    reviewers: List[str]  # Relecteurs

    # Organisation
    department: Optional[str]  # Service/département
    business_unit: Optional[str]  # Unité d'affaires
    geographic_scope: List[str]  # Portée géographique

    # Relations
    supersedes: List[str]  # Documents remplacés
    superseded_by: Optional[str]  # Remplacé par
    related_documents: List[str]  # Documents liés
    parent_document: Optional[str]  # Document parent
    child_documents: List[str]  # Documents enfants

    # Conformité et audit
    compliance_mappings: Dict[str, List[str]] = field(default_factory=dict)  # Framework -> Requirements
    audit_trail: List[Dict] = field(default_factory=list)  # Historique des modifications
    control_references: List[str] = field(default_factory=list)  # Références aux contrôles
    evidence_for: List[str] = field(default_factory=list)  # Preuve pour quels contrôles

    # Contexte métier
    business_processes: List[str] = field(default_factory=list)  # Processus métier concernés
    systems_affected: List[str] = field(default_factory=list)  # Systèmes impactés
    data_categories: List[str] = field(default_factory=list)  # Catégories de données

    # Qualité et fiabilité
    completeness_score: float = 0.0  # Score de complétude
    quality_indicators: Dict[str, Any] = field(default_factory=dict)  # Indicateurs qualité
    validation_status: str = "pending"  # Status de validation


class MetadataExtractor:
    """
    Extracteur avancé de métadonnées pour documents de cybersécurité.

    Cette classe va au-delà de l'extraction basique pour comprendre
    le contexte métier et réglementaire des documents. Elle identifie :
    - Les relations entre documents
    - Le cycle de vie documentaire
    - Les mappings de conformité
    - Les responsabilités et approbations
    - La classification et sensibilité

    C'est comme un archiviste expert qui catalogue chaque document
    avec toutes ses caractéristiques importantes.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'extracteur avec sa configuration.

        Args:
            config: Configuration incluant :
                - organization_name: Nom de l'organisation
                - default_classification: Classification par défaut
                - extract_pii: Détecter les données personnelles
                - validate_references: Valider les références croisées
                - quality_checks: Activer les contrôles qualité
        """
        self.config = config or {}

        # Configuration
        self.organization_name = self.config.get('organization_name', 'Organization')
        self.default_classification = self.config.get('default_classification', 'Internal')
        self.extract_pii = self.config.get('extract_pii', True)
        self.validate_references = self.config.get('validate_references', True)
        self.quality_checks = self.config.get('quality_checks', True)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialiser les patterns et bases de connaissances
        self._init_patterns()
        self._init_knowledge_bases()

        # Cache pour les validations
        self.document_registry = {}  # Pour valider les références

    def _init_patterns(self):
        """
        Initialise les patterns regex pour l'extraction.
        """
        # Pattern pour les codes de documents
        self.doc_code_patterns = [
            re.compile(r'\b(POL|PROC|STD|GUI|REF|FORM)-([A-Z]{2,})-(\d{3,})\b'),  # POL-SEC-001
            re.compile(r'\b([A-Z]{2,})-(\d{4})-(\d{2})\b'),  # ISO-27001-01
            re.compile(r'\bDocument\s*(?:Code|Ref|#)\s*:\s*([A-Z0-9\-\.]+)\b', re.IGNORECASE),
            re.compile(r'\bRéférence\s*:\s*([A-Z0-9\-\.]+)\b', re.IGNORECASE)
        ]

        # Pattern pour les versions
        self.version_patterns = [
            re.compile(r'\bVersion\s*:?\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            re.compile(r'\bV(\d+(?:\.\d+)*)\b'),
            re.compile(r'\bRev(?:ision)?\s*:?\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            re.compile(r'\b(\d+(?:\.\d+)*)\s*\|\s*(?:Draft|Final|Approved)\b', re.IGNORECASE)
        ]

        # Pattern pour les dates
        self.date_patterns = {
            'creation': [
                re.compile(r'(?:Créé|Created|Établi)\s*(?:le|on)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                           re.IGNORECASE),
                re.compile(r'Date\s*de\s*création\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE)
            ],
            'modification': [
                re.compile(r'(?:Modifié|Modified|Mis à jour)\s*(?:le|on)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                           re.IGNORECASE),
                re.compile(r'Dernière\s*modification\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE)
            ],
            'review': [
                re.compile(r'(?:Revu|Reviewed|À revoir)\s*(?:le|on|avant)?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                           re.IGNORECASE),
                re.compile(r'Prochaine\s*(?:revue|révision)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE)
            ],
            'expiry': [
                re.compile(r'(?:Expire|Valide jusqu\'au)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE),
                re.compile(r'Date\s*d\'expiration\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE)
            ]
        }

        # Pattern pour les personnes
        self.people_patterns = {
            'author': [
                re.compile(r'(?:Auteur|Author|Rédigé par|Written by)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE),
                re.compile(r'(?:Préparé par|Prepared by)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE)
            ],
            'owner': [
                re.compile(r'(?:Propriétaire|Owner|Responsable)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE),
                re.compile(r'(?:Document Owner)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE)
            ],
            'approver': [
                re.compile(r'(?:Approuvé par|Approved by|Validé par)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE),
                re.compile(r'(?:Approbateur|Approver)\s*:?\s*([A-Za-zÀ-ÿ\s\-\.]+)', re.IGNORECASE)
            ]
        }

        # Pattern pour la classification
        self.classification_patterns = [
            re.compile(
                r'(?:Classification|Confidentialité)\s*:?\s*(Public|Interne|Internal|Confidentiel|Confidential|Secret)',
                re.IGNORECASE),
            re.compile(r'\b(PUBLIC|INTERNE|INTERNAL|CONFIDENTIEL|CONFIDENTIAL|SECRET)\b'),
            re.compile(r'(?:Diffusion|Distribution)\s*:?\s*(Restreinte|Restricted|Limitée|Limited)', re.IGNORECASE)
        ]

        # Pattern pour les données sensibles (PII/GDPR)
        self.pii_patterns = {
            'personal_data': [
                re.compile(r'\b(?:données personnelles|personal data|DCP)\b', re.IGNORECASE),
                re.compile(r'\b(?:nom|prénom|name|surname)\b', re.IGNORECASE),
                re.compile(r'\b(?:adresse|address)\b', re.IGNORECASE),
                re.compile(r'\b(?:email|courriel|e-mail)\b', re.IGNORECASE),
                re.compile(r'\b(?:téléphone|phone|mobile)\b', re.IGNORECASE)
            ],
            'sensitive_data': [
                re.compile(r'\b(?:santé|health|médical)\b', re.IGNORECASE),
                re.compile(r'\b(?:financier|financial|bancaire|banking)\b', re.IGNORECASE),
                re.compile(r'\b(?:judiciaire|judicial|pénal|criminal)\b', re.IGNORECASE),
                re.compile(r'\b(?:biométrique|biometric)\b', re.IGNORECASE)
            ],
            'identifiers': [
                re.compile(r'\b(?:NIR|numéro de sécurité sociale|SSN)\b', re.IGNORECASE),
                re.compile(r'\b(?:CNI|carte d\'identité|ID card)\b', re.IGNORECASE),
                re.compile(r'\b(?:passeport|passport)\b', re.IGNORECASE)
            ]
        }

        # Pattern pour les systèmes et processus
        self.system_patterns = [
            re.compile(r'(?:Système|System|Application)\s*:?\s*([A-Za-z0-9\-_\s]+)', re.IGNORECASE),
            re.compile(r'(?:Concerne|Affects|Impacte)\s*:?\s*([A-Za-z0-9\-_\s]+)', re.IGNORECASE),
            re.compile(r'\b([A-Z]{2,}(?:[_\-][A-Z0-9]+)*)\b')  # Acronymes type ERP_PROD
        ]

        # Pattern pour les processus métier
        self.process_patterns = [
            re.compile(r'(?:Processus|Process)\s*:?\s*([A-Za-zÀ-ÿ\s\-]+)', re.IGNORECASE),
            re.compile(r'(?:Procédure|Procedure)\s+de\s+([A-Za-zÀ-ÿ\s\-]+)', re.IGNORECASE)
        ]

    def _init_knowledge_bases(self):
        """
        Initialise les bases de connaissances métier.
        """
        # Types de documents reconnus
        self.document_types = {
            'POL': 'Policy',
            'PROC': 'Procedure',
            'STD': 'Standard',
            'GUI': 'Guideline',
            'REF': 'Reference',
            'FORM': 'Form',
            'RPT': 'Report',
            'PLAN': 'Plan',
            'TEMP': 'Template'
        }

        # Mapping des classifications
        self.classification_mapping = {
            'public': 'Public',
            'interne': 'Internal',
            'internal': 'Internal',
            'confidentiel': 'Confidential',
            'confidential': 'Confidential',
            'secret': 'Secret',
            'restreint': 'Restricted',
            'restricted': 'Restricted'
        }

        # Départements/services courants
        self.departments = [
            'IT', 'SI', 'DSI', 'Informatique',
            'Security', 'Sécurité', 'SSI',
            'Legal', 'Juridique',
            'HR', 'RH', 'Ressources Humaines',
            'Finance', 'Comptabilité',
            'Operations', 'Opérations',
            'Risk', 'Risques',
            'Compliance', 'Conformité',
            'Audit'
        ]

        # Indicateurs de qualité
        self.quality_indicators = {
            'has_version': 0.1,
            'has_date': 0.1,
            'has_author': 0.1,
            'has_approver': 0.15,
            'has_classification': 0.15,
            'has_references': 0.1,
            'has_doc_code': 0.2,
            'has_owner': 0.1
        }

        # Stades du cycle de vie
        self.lifecycle_stages = {
            'draft': ['draft', 'brouillon', 'ébauche', 'version de travail'],
            'review': ['en révision', 'under review', 'en relecture', 'à valider'],
            'approved': ['approuvé', 'approved', 'validé', 'final'],
            'active': ['actif', 'active', 'en vigueur', 'applicable'],
            'obsolete': ['obsolète', 'obsolete', 'périmé', 'deprecated', 'remplacé']
        }

    def extract_metadata(self, parsed_doc: ParsedDocument) -> DocumentMetadata:
        """
        Extrait toutes les métadonnées du document.

        Cette méthode orchestre l'extraction complète des métadonnées
        en appliquant tous les extracteurs spécialisés.

        Args:
            parsed_doc: Document parsé

        Returns:
            DocumentMetadata avec toutes les informations extraites
        """
        self.logger.info(f"Extraction des métadonnées : {parsed_doc.metadata.get('file_name', 'Unknown')}")

        # Initialiser les métadonnées
        metadata = DocumentMetadata(
            document_id=self._generate_document_id(parsed_doc),
            document_code=None,
            version=None,
            revision=None,
            classification_level=self.default_classification,
            data_sensitivity=[],
            handling_restrictions=[],
            creation_date=None,
            last_modified=None,
            review_date=None,
            expiry_date=None,
            lifecycle_stage='unknown',
            author=None,
            contributors=[],
            owner=None,
            approver=None,
            reviewers=[],
            department=None,
            business_unit=None,
            geographic_scope=[],
            supersedes=[],
            superseded_by=None,
            related_documents=[],
            parent_document=None,
            child_documents=[]
        )

        # Extraire les différentes catégories de métadonnées
        self._extract_identification(parsed_doc, metadata)
        self._extract_dates(parsed_doc, metadata)
        self._extract_people(parsed_doc, metadata)
        self._extract_classification(parsed_doc, metadata)
        self._extract_organization(parsed_doc, metadata)
        self._extract_relationships(parsed_doc, metadata)
        self._extract_compliance_mappings(parsed_doc, metadata)
        self._extract_business_context(parsed_doc, metadata)

        # Détecter les données sensibles si demandé
        if self.extract_pii:
            self._detect_sensitive_data(parsed_doc, metadata)

        # Calculer les indicateurs de qualité
        if self.quality_checks:
            self._calculate_quality_scores(metadata)

        # Déterminer le stade du cycle de vie
        metadata.lifecycle_stage = self._determine_lifecycle_stage(parsed_doc, metadata)

        # Ajouter les métadonnées au document
        self._enrich_document_metadata(parsed_doc, metadata)

        self.logger.info(
            f"Métadonnées extraites : "
            f"Code={metadata.document_code}, "
            f"Version={metadata.version}, "
            f"Classification={metadata.classification_level}"
        )

        return metadata

    def _generate_document_id(self, parsed_doc: ParsedDocument) -> str:
        """
        Génère un ID unique pour le document.
        """
        # Utiliser le hash du document s'il existe
        if parsed_doc.document_hash:
            return f"DOC-{parsed_doc.document_hash[:12]}"

        # Sinon, générer un hash basé sur le contenu et le chemin
        content_hash = hashlib.sha256(
            f"{parsed_doc.source_path}{parsed_doc.content[:1000]}".encode()
        ).hexdigest()

        return f"DOC-{content_hash[:12]}"

    def _extract_identification(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait les informations d'identification du document.
        """
        content = parsed_doc.content

        # Chercher le code du document
        for pattern in self.doc_code_patterns:
            match = pattern.search(content[:2000])  # Chercher dans l'en-tête
            if match:
                if match.lastindex and match.lastindex >= 3:
                    # Format structuré TYPE-DOMAIN-NUMBER
                    doc_type = match.group(1)
                    domain = match.group(2)
                    number = match.group(3)
                    metadata.document_code = f"{doc_type}-{domain}-{number}"
                else:
                    # Format libre
                    metadata.document_code = match.group(1) if match.lastindex else match.group(0)
                break

        # Chercher la version
        for pattern in self.version_patterns:
            match = pattern.search(content[:2000])
            if match:
                metadata.version = match.group(1)
                break

        # Si pas de version trouvée, chercher dans les métadonnées du fichier
        if not metadata.version:
            metadata.version = parsed_doc.metadata.get('version', '1.0')

    def _extract_dates(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait toutes les dates importantes du document.
        """
        content = parsed_doc.content[:5000]  # Chercher dans l'en-tête

        for date_type, patterns in self.date_patterns.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    date_str = match.group(1)
                    parsed_date = self._parse_date(date_str)

                    if parsed_date:
                        if date_type == 'creation':
                            metadata.creation_date = parsed_date
                        elif date_type == 'modification':
                            metadata.last_modified = parsed_date
                        elif date_type == 'review':
                            metadata.review_date = parsed_date
                        elif date_type == 'expiry':
                            metadata.expiry_date = parsed_date
                    break

        # Utiliser les dates du système de fichiers si pas trouvées
        if not metadata.creation_date:
            if 'created_at' in parsed_doc.metadata:
                try:
                    metadata.creation_date = datetime.fromisoformat(
                        parsed_doc.metadata['created_at']
                    ).date()
                except:
                    pass

        if not metadata.last_modified:
            if 'modified_at' in parsed_doc.metadata:
                try:
                    metadata.last_modified = datetime.fromisoformat(
                        parsed_doc.metadata['modified_at']
                    ).date()
                except:
                    pass

    def _parse_date(self, date_str: str) -> Optional[date]:
        """
        Parse une chaîne de date en objet date.
        """
        # Nettoyer la chaîne
        date_str = date_str.strip()

        # Essayer différents formats
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%d/%m/%y',
            '%d-%m-%y',
            '%Y-%m-%d',
            '%Y/%m/%d'
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                # Gérer les années à 2 chiffres
                if parsed.year < 100:
                    if parsed.year > 50:
                        parsed = parsed.replace(year=1900 + parsed.year)
                    else:
                        parsed = parsed.replace(year=2000 + parsed.year)
                return parsed.date()
            except ValueError:
                continue

        return None

    def _extract_people(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait les personnes mentionnées dans le document.
        """
        content = parsed_doc.content[:5000]

        for role, patterns in self.people_patterns.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    person = match.group(1).strip()
                    # Nettoyer le nom
                    person = re.sub(r'\s+', ' ', person)
                    person = person.strip('.,;:')

                    if len(person) > 3 and len(person) < 50:  # Validation basique
                        if role == 'author':
                            metadata.author = person
                        elif role == 'owner':
                            metadata.owner = person
                        elif role == 'approver':
                            metadata.approver = person
                    break

        # Chercher les contributeurs/relecteurs dans les sections spécifiques
        contributors_section = re.search(
            r'(?:Contributeurs|Contributors|Participants)\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
            content,
            re.IGNORECASE
        )

        if contributors_section:
            contributors_text = contributors_section.group(1)
            # Extraire les noms (supposer qu'ils sont séparés par des virgules ou des sauts de ligne)
            names = re.split(r'[,\n;]', contributors_text)
            for name in names:
                name = name.strip('.,;:- ')
                if 3 < len(name) < 50 and not name.lower().startswith(('et ', 'and ')):
                    metadata.contributors.append(name)

    def _extract_classification(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait la classification et les restrictions du document.
        """
        content = parsed_doc.content[:2000]

        # Chercher la classification
        for pattern in self.classification_patterns:
            match = pattern.search(content)
            if match:
                classification = match.group(1).lower()
                if classification in self.classification_mapping:
                    metadata.classification_level = self.classification_mapping[classification]
                    break

        # Chercher les restrictions de manipulation
        if 'restreint' in content.lower() or 'restricted' in content.lower():
            metadata.handling_restrictions.append('Distribution restreinte')

        if 'ne pas diffuser' in content.lower() or 'do not distribute' in content.lower():
            metadata.handling_restrictions.append('Ne pas diffuser')

        if 'usage interne' in content.lower() or 'internal use only' in content.lower():
            metadata.handling_restrictions.append('Usage interne uniquement')

        # Marquer si c'est un document de sécurité
        if any(word in parsed_doc.doc_type.lower() for word in ['security', 'sécurité', 'policy', 'politique']):
            if metadata.classification_level == 'Public':
                metadata.classification_level = 'Internal'  # Minimum pour les docs de sécurité

    def _extract_organization(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait les informations organisationnelles.
        """
        content = parsed_doc.content[:3000]

        # Chercher le département
        for dept in self.departments:
            if re.search(r'\b' + re.escape(dept) + r'\b', content, re.IGNORECASE):
                metadata.department = dept
                break

        # Chercher la portée géographique
        geo_patterns = [
            r'(?:Applicable à|Applies to|Concerne)\s*:?\s*([^\n]+)',
            r'(?:Portée|Scope|Périmètre)\s*:?\s*([^\n]+)',
            r'(?:Sites?|Locations?|Établissements?)\s*:?\s*([^\n]+)'
        ]

        for pattern in geo_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                scope_text = match.group(1)
                # Extraire les lieux
                locations = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', scope_text)
                metadata.geographic_scope.extend(locations[:5])  # Limiter
                break

    def _extract_relationships(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait les relations avec d'autres documents.
        """
        content = parsed_doc.content

        # Patterns pour les relations
        relationship_patterns = {
            'supersedes': [
                r'(?:Remplace|Supersedes|Annule et remplace)\s*:?\s*([A-Z0-9\-\.]+)',
                r'(?:Ce document remplace|This document supersedes)\s+([A-Z0-9\-\.]+)'
            ],
            'superseded_by': [
                r'(?:Remplacé par|Superseded by|Obsolète, voir)\s*:?\s*([A-Z0-9\-\.]+)'
            ],
            'related': [
                r'(?:Voir aussi|See also|Documents? liés?|Related documents?)\s*:?\s*([^\n]+)',
                r'(?:Référence|Reference)\s*:?\s*([A-Z0-9\-\.]+)'
            ]
        }

        # Extraire les relations
        for rel_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    ref = match.group(1).strip()

                    # Nettoyer et valider la référence
                    if re.match(r'^[A-Z0-9\-\.]+$', ref) and len(ref) < 30:
                        if rel_type == 'supersedes':
                            metadata.supersedes.append(ref)
                        elif rel_type == 'superseded_by' and not metadata.superseded_by:
                            metadata.superseded_by = ref
                        elif rel_type == 'related':
                            metadata.related_documents.append(ref)

        # Déduplication
        metadata.supersedes = list(set(metadata.supersedes))
        metadata.related_documents = list(set(metadata.related_documents))

        # Valider les références si demandé
        if self.validate_references and self.document_registry:
            self._validate_document_references(metadata)

    def _extract_compliance_mappings(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait les mappings vers les frameworks de conformité.
        """
        content = parsed_doc.content

        # Patterns pour les mappings de conformité
        compliance_patterns = {
            'ISO 27001': r'ISO\s*27001[\s:]*(?:(?:A\.)?(\d+(?:\.\d+)*))',
            'ISO 27002': r'ISO\s*27002[\s:]*(?:(\d+(?:\.\d+)*))',
            'NIST': r'NIST\s+([A-Z]{2}\-\d+(?:\.\d+)*)',
            'PCI DSS': r'PCI[\s\-]DSS\s+(?:Requirement\s+)?(\d+(?:\.\d+)*)',
            'RGPD': r'(?:RGPD|GDPR)\s+Article\s+(\d+)',
            'SOC 2': r'SOC\s*2\s+(?:TSC\s+)?([A-Z]{2}\d+(?:\.\d+)*)'
        }

        # Extraire les mappings
        for framework, pattern in compliance_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            requirements = []

            for match in matches:
                if match.lastindex:
                    req = match.group(1)
                    requirements.append(req)

            if requirements:
                metadata.compliance_mappings[framework] = list(set(requirements))

        # Chercher aussi les références aux contrôles internes
        control_pattern = re.compile(r'(?:Contrôle|Control)\s+([A-Z]{2,}\-\d+)', re.IGNORECASE)
        controls = []

        for match in control_pattern.finditer(content):
            controls.append(match.group(1))

        if controls:
            metadata.control_references = list(set(controls))

    def _extract_business_context(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Extrait le contexte métier du document.
        """
        content = parsed_doc.content

        # Extraire les systèmes affectés
        for pattern in self.system_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                system = match.group(1).strip()
                # Valider que c'est probablement un système
                if (len(system) > 2 and
                        not system.lower() in ['le', 'la', 'les', 'de', 'et', 'ou'] and
                        (system.isupper() or '_' in system or '-' in system)):
                    metadata.systems_affected.append(system)

        # Limiter et dédupliquer
        metadata.systems_affected = list(set(metadata.systems_affected[:10]))

        # Extraire les processus métier
        for pattern in self.process_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                process = match.group(1).strip()
                if 5 < len(process) < 50:
                    metadata.business_processes.append(process)

        metadata.business_processes = list(set(metadata.business_processes[:10]))

        # Détecter les catégories de données
        data_categories = {
            'Données personnelles': ['données personnelles', 'personal data', 'PII'],
            'Données financières': ['financier', 'financial', 'bancaire', 'paiement'],
            'Données de santé': ['santé', 'health', 'médical', 'patient'],
            'Données techniques': ['logs', 'configuration', 'système', 'infrastructure'],
            'Données métier': ['client', 'contrat', 'commande', 'produit']
        }

        for category, keywords in data_categories.items():
            if any(keyword.lower() in content.lower() for keyword in keywords):
                metadata.data_categories.append(category)

    def _detect_sensitive_data(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Détecte la présence de données sensibles dans le document.
        """
        content = parsed_doc.content.lower()

        # Vérifier chaque catégorie de données sensibles
        for category, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    if category == 'personal_data':
                        metadata.data_sensitivity.append('Données personnelles')
                    elif category == 'sensitive_data':
                        metadata.data_sensitivity.append('Données sensibles')
                    elif category == 'identifiers':
                        metadata.data_sensitivity.append('Identifiants personnels')
                    break

        # Déduplication
        metadata.data_sensitivity = list(set(metadata.data_sensitivity))

        # Ajuster la classification si nécessaire
        if metadata.data_sensitivity and metadata.classification_level == 'Public':
            metadata.classification_level = 'Internal'
            metadata.handling_restrictions.append('Contient des données sensibles')

    def _calculate_quality_scores(self, metadata: DocumentMetadata):
        """
        Calcule les scores de qualité du document.
        """
        score = 0.0
        indicators = {}

        # Vérifier chaque indicateur
        if metadata.version:
            score += self.quality_indicators['has_version']
            indicators['has_version'] = True

        if metadata.creation_date or metadata.last_modified:
            score += self.quality_indicators['has_date']
            indicators['has_date'] = True

        if metadata.author:
            score += self.quality_indicators['has_author']
            indicators['has_author'] = True

        if metadata.approver:
            score += self.quality_indicators['has_approver']
            indicators['has_approver'] = True

        if metadata.classification_level != self.default_classification:
            score += self.quality_indicators['has_classification']
            indicators['has_classification'] = True

        if metadata.related_documents or metadata.supersedes:
            score += self.quality_indicators['has_references']
            indicators['has_references'] = True

        if metadata.document_code:
            score += self.quality_indicators['has_doc_code']
            indicators['has_doc_code'] = True

        if metadata.owner:
            score += self.quality_indicators['has_owner']
            indicators['has_owner'] = True

        metadata.completeness_score = score
        metadata.quality_indicators = indicators

        # Déterminer le statut de validation
        if score >= 0.8:
            metadata.validation_status = 'validated'
        elif score >= 0.5:
            metadata.validation_status = 'partial'
        else:
            metadata.validation_status = 'incomplete'

    def _determine_lifecycle_stage(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata) -> str:
        """
        Détermine le stade du cycle de vie du document.
        """
        content = parsed_doc.content.lower()

        # Vérifier les indicateurs de stade
        for stage, indicators in self.lifecycle_stages.items():
            for indicator in indicators:
                if indicator in content:
                    return stage

        # Logique basée sur d'autres métadonnées
        if metadata.superseded_by:
            return 'obsolete'

        if metadata.approver and metadata.approval_date:
            return 'active'

        if metadata.version and 'draft' in metadata.version.lower():
            return 'draft'

        if metadata.review_date and metadata.review_date > date.today():
            return 'active'

        return 'unknown'

    def _validate_document_references(self, metadata: DocumentMetadata):
        """
        Valide les références à d'autres documents.
        """
        # Valider que les documents référencés existent
        for ref in metadata.related_documents[:]:
            if ref not in self.document_registry:
                self.logger.warning(f"Référence invalide : {ref}")
                metadata.related_documents.remove(ref)

        # Vérifier la cohérence des relations
        if metadata.superseded_by and metadata.lifecycle_stage != 'obsolete':
            self.logger.warning(
                f"Document {metadata.document_code} remplacé mais pas marqué obsolète"
            )

    def _enrich_document_metadata(self, parsed_doc: ParsedDocument, metadata: DocumentMetadata):
        """
        Enrichit les métadonnées du document parsé.
        """
        # Ajouter toutes les métadonnées extraites
        parsed_doc.metadata['extracted_metadata'] = {
            'document_id': metadata.document_id,
            'document_code': metadata.document_code,
            'version': metadata.version,
            'classification': metadata.classification_level,
            'lifecycle_stage': metadata.lifecycle_stage,
            'author': metadata.author,
            'owner': metadata.owner,
            'department': metadata.department,
            'compliance_mappings': metadata.compliance_mappings,
            'quality_score': metadata.completeness_score,
            'validation_status': metadata.validation_status,
            'data_sensitivity': metadata.data_sensitivity,
            'systems_affected': metadata.systems_affected,
            'business_processes': metadata.business_processes
        }

        # Ajouter des flags pour la recherche
        parsed_doc.metadata['has_compliance_mappings'] = bool(metadata.compliance_mappings)
        parsed_doc.metadata['has_sensitive_data'] = bool(metadata.data_sensitivity)
        parsed_doc.metadata['is_validated'] = metadata.validation_status == 'validated'
        parsed_doc.metadata['is_current'] = metadata.lifecycle_stage in ['active', 'approved']

    def register_document(self, doc_code: str, doc_info: Dict):
        """
        Enregistre un document dans le registre pour validation des références.
        """
        if doc_code:
            self.document_registry[doc_code] = doc_info

    def get_document_registry(self) -> Dict[str, Dict]:
        """
        Retourne le registre des documents connus.
        """
        return self.document_registry.copy()
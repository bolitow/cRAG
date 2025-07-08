# src/ingestion/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class ParsedDocument:
    """
    Représente un document parsé avec toutes ses métadonnées.

    C'est comme une fiche d'identité complète du document qui contient
    non seulement son contenu, mais aussi toutes les informations contextuelles
    nécessaires pour un traitement intelligent.
    """
    content: str  # Le contenu textuel extrait
    metadata: Dict  # Métadonnées enrichies
    source_path: str  # Chemin d'origine du fichier
    doc_type: str  # Type identifié (policy, procedure, etc.)
    sections: List[Dict]  # Sections identifiées dans le document
    tables: List[Dict]  # Tableaux extraits (si applicable)
    images_count: int  # Nombre d'images (pour info)
    parsing_errors: List[str]  # Erreurs rencontrées lors du parsing
    parsing_timestamp: datetime  # Quand le document a été parsé
    document_hash: str  # Hash pour détecter les modifications
    confidence_score: float  # Confiance dans la qualité du parsing


class BaseParser(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les parsers.

    Cette classe est comme un contrat que chaque parser spécialisé doit respecter.
    Elle garantit que peu importe le type de document, le système pourra toujours
    l'utiliser de la même manière.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le parser avec une configuration optionnelle.

        Args:
            config: Configuration spécifique au parser
                   (ex: langue, encoding, options d'extraction)
        """
        self.config = config or {}
        self.supported_extensions = []  # À définir dans chaque sous-classe
        self.parsing_stats = {
            "total_parsed": 0,
            "successful": 0,
            "failed": 0,
            "warnings": 0
        }

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """
        Vérifie si ce parser peut traiter le fichier donné.

        C'est comme demander "Parles-tu cette langue ?"

        Args:
            file_path: Chemin vers le fichier à vérifier

        Returns:
            True si le parser peut traiter ce fichier
        """
        pass

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse le document et extrait toutes les informations pertinentes.

        C'est le cœur du parser : transformer le document brut en données structurées.

        Args:
            file_path: Chemin vers le fichier à parser

        Returns:
            ParsedDocument contenant toutes les données extraites

        Raises:
            ParsingError: Si le parsing échoue
        """
        pass

    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extrait les métadonnées de base du fichier.

        Cette méthode commune extrait les informations que tous les fichiers
        possèdent : taille, date de création, etc.
        """
        import os
        from datetime import datetime

        stat = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_extension": os.path.splitext(file_path)[1].lower()
        }

    def calculate_document_hash(self, content: str) -> str:
        """
        Calcule un hash unique du contenu pour détecter les changements.

        C'est comme une empreinte digitale du document qui change
        si même un seul caractère est modifié.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def identify_document_type(self, content: str, metadata: Dict) -> str:
        """
        Détermine le type de document basé sur son contenu et ses métadonnées.

        Cette méthode utilise des heuristiques pour classifier le document
        (politique, procédure, rapport d'audit, etc.)
        """
        content_lower = content.lower()[:2000]  # Analyser le début

        # Analyse basée sur des mots-clés caractéristiques
        type_indicators = {
            "policy": [
                "politique de sécurité", "security policy", "politique générale",
                "cette politique", "la présente politique", "policy statement"
            ],
            "procedure": [
                "procédure", "procedure", "mode opératoire", "instructions",
                "étapes à suivre", "process steps", "marche à suivre"
            ],
            "standard": [
                "standard", "norme", "spécifications", "requirements",
                "exigences techniques", "technical standard"
            ],
            "guideline": [
                "guide", "guideline", "recommandations", "bonnes pratiques",
                "best practices", "lignes directrices"
            ],
            "audit_report": [
                "rapport d'audit", "audit report", "findings", "constatations",
                "non-conformités", "plan d'action", "corrective actions"
            ],
            "risk_assessment": [
                "évaluation des risques", "risk assessment", "analyse de risque",
                "matrice des risques", "risk matrix", "impact analysis"
            ],
            "incident_report": [
                "rapport d'incident", "incident report", "security incident",
                "breach report", "violation de sécurité"
            ],
            "faq": [
                "faq", "questions fréquentes", "frequently asked", "q&a",
                "questions et réponses", "questions/réponses"
            ],
            "control_matrix": [
                "matrice de contrôle", "control matrix", "liste des contrôles",
                "control framework", "cadre de contrôle"
            ]
        }

        # Compter les occurrences de chaque type
        type_scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                type_scores[doc_type] = score

        # Vérifier aussi le nom du fichier
        file_name = metadata.get("file_name", "").lower()
        for doc_type, indicators in type_indicators.items():
            for indicator in indicators:
                if indicator.replace(" ", "_") in file_name or indicator.replace(" ", "-") in file_name:
                    type_scores[doc_type] = type_scores.get(doc_type, 0) + 2

        # Retourner le type avec le score le plus élevé
        if type_scores:
            return max(type_scores, key=type_scores.get)

        return "general"

    def extract_sections(self, content: str) -> List[Dict]:
        """
        Extrait les sections principales du document.

        Cette méthode générique identifie les grandes parties du document
        basées sur des patterns communs (titres, numérotation, etc.)
        """
        sections = []

        # Patterns pour détecter les titres de section
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(\d+\.(?:\d+\.)*)\s+(.+)$',  # Numérotation hiérarchique
            r'^([A-Z][A-Z\s]+)$',  # Titres en majuscules
            r'^(Article|Section|Chapitre)\s+\d+\s*:\s*(.+)$',  # Structures formelles
        ]

        lines = content.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Vérifier si c'est un titre de section
            is_section_title = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Sauvegarder la section précédente
                    if current_section:
                        sections.append(current_section)

                    # Commencer une nouvelle section
                    current_section = {
                        "title": line,
                        "level": self._determine_section_level(line),
                        "start_line": i,
                        "content": ""
                    }
                    is_section_title = True
                    break

            # Si ce n'est pas un titre, ajouter au contenu de la section courante
            if not is_section_title and current_section:
                current_section["content"] += line + "\n"

        # Ajouter la dernière section
        if current_section:
            sections.append(current_section)

        return sections

    def _determine_section_level(self, title: str) -> int:
        """
        Détermine le niveau hiérarchique d'une section.

        Plus le niveau est bas, plus la section est importante
        (niveau 1 = titre principal, niveau 2 = sous-titre, etc.)
        """
        # Compter les # pour Markdown
        if title.startswith('#'):
            return len(title) - len(title.lstrip('#'))

        # Compter les points pour la numérotation
        if re.match(r'^\d+\.', title):
            return title.count('.') + 1

        # Titres formels
        if title.startswith(('Article', 'Chapitre')):
            return 1
        elif title.startswith('Section'):
            return 2

        # Titre en majuscules = niveau 1
        if title.isupper() and len(title) > 3:
            return 1

        return 3  # Niveau par défaut

    def validate_parsing_result(self, parsed_doc: ParsedDocument) -> Tuple[bool, List[str]]:
        """
        Valide la qualité du parsing et retourne les avertissements éventuels.

        Cette méthode vérifie que le document a été correctement parsé
        et identifie les problèmes potentiels.
        """
        warnings = []

        # Vérifier que le contenu n'est pas vide
        if not parsed_doc.content or len(parsed_doc.content.strip()) < 100:
            warnings.append("Le contenu extrait semble très court ou vide")

        # Vérifier la présence de caractères étranges (mauvais encoding)
        strange_chars = sum(1 for c in parsed_doc.content if ord(c) > 65535)
        if strange_chars > len(parsed_doc.content) * 0.01:  # Plus de 1% de caractères étranges
            warnings.append("Possible problème d'encoding détecté")

        # Vérifier que des sections ont été trouvées pour les documents structurés
        if parsed_doc.doc_type in ["policy", "procedure", "standard"] and not parsed_doc.sections:
            warnings.append(f"Aucune section trouvée dans ce document de type {parsed_doc.doc_type}")

        # Vérifier le score de confiance
        if parsed_doc.confidence_score < 0.7:
            warnings.append(f"Score de confiance faible : {parsed_doc.confidence_score:.2f}")

        # Le parsing est valide s'il n'y a pas d'erreurs critiques
        is_valid = len(parsed_doc.parsing_errors) == 0 and parsed_doc.content

        return is_valid, warnings

    def update_stats(self, success: bool, warnings_count: int = 0):
        """
        Met à jour les statistiques de parsing.

        Cela permet de surveiller la performance et la fiabilité du parser.
        """
        self.parsing_stats["total_parsed"] += 1
        if success:
            self.parsing_stats["successful"] += 1
        else:
            self.parsing_stats["failed"] += 1
        self.parsing_stats["warnings"] += warnings_count

    def get_stats_summary(self) -> str:
        """
        Retourne un résumé des statistiques de parsing.
        """
        total = self.parsing_stats["total_parsed"]
        if total == 0:
            return "Aucun document parsé"

        success_rate = (self.parsing_stats["successful"] / total) * 100
        return (f"Documents parsés : {total} | "
                f"Succès : {success_rate:.1f}% | "
                f"Avertissements : {self.parsing_stats['warnings']}")


class ParsingError(Exception):
    """Exception levée quand le parsing échoue."""
    pass


import re
# src/ingestion/document_parser.py
import os
import mimetypes
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json

# Import des parsers spécialisés
from .parsers.base_parser import BaseParser, ParsedDocument, ParsingError
from .parsers.pdf_parser import PDFParser
from .parsers.markdown_parser import MarkdownParser
from .parsers.excel_parser import ExcelParser
from .parsers.faq_parser import FAQParser


class DocumentParser:
    """
    Orchestrateur principal pour le parsing de documents.

    Cette classe est le point d'entrée unique pour parser tous types de documents.
    Elle :
    - Détecte automatiquement le type de document
    - Sélectionne le parser approprié
    - Gère le parsing en parallèle pour les performances
    - Maintient un cache des documents parsés
    - Génère des rapports de parsing

    C'est comme un chef d'orchestre qui dirige différents musiciens (parsers)
    selon la partition (type de document).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'orchestrateur de parsing.

        Args:
            config: Configuration globale incluant :
                - cache_enabled: Activer le cache des documents parsés
                - cache_dir: Répertoire pour le cache
                - parallel_processing: Activer le traitement parallèle
                - max_workers: Nombre max de workers parallèles
                - file_size_limit: Limite de taille des fichiers (MB)
                - auto_detect_type: Détection automatique du type
                - parser_configs: Configurations spécifiques par parser
        """
        self.config = config or {}

        # Configuration du cache
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_dir = Path(self.config.get('cache_dir', '.document_cache'))
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)

        # Configuration du traitement parallèle
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 4)

        # Limites
        self.file_size_limit = self.config.get('file_size_limit', 100) * 1024 * 1024  # MB to bytes

        # Configuration de détection
        self.auto_detect_type = self.config.get('auto_detect_type', True)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialiser les parsers disponibles
        self._init_parsers()

        # Statistiques de parsing
        self.parsing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'by_type': {},
            'total_time': 0.0
        }

        # Registre des erreurs
        self.error_log = []

    def _init_parsers(self):
        """
        Initialise tous les parsers disponibles avec leurs configurations.

        Chaque parser peut avoir sa propre configuration spécifique.
        """
        parser_configs = self.config.get('parser_configs', {})

        self.parsers = {
            'pdf': PDFParser(parser_configs.get('pdf', {})),
            'markdown': MarkdownParser(parser_configs.get('markdown', {})),
            'excel': ExcelParser(parser_configs.get('excel', {})),
            'faq': FAQParser(parser_configs.get('faq', {}))
        }

        # Mapping des extensions aux types de parser
        self.extension_mapping = {
            '.pdf': 'pdf',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.mdown': 'markdown',
            '.mkd': 'markdown',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.xlsm': 'excel',
            '.xlsb': 'excel',
            '.faq': 'faq',
            '.txt': 'faq',  # Peut être FAQ ou autre
            '.html': 'faq',  # Peut être FAQ ou autre
            '.htm': 'faq'
        }

        # Types MIME supportés
        self.mime_mapping = {
            'application/pdf': 'pdf',
            'text/markdown': 'markdown',
            'text/x-markdown': 'markdown',
            'application/vnd.ms-excel': 'excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
            'text/plain': 'faq',
            'text/html': 'faq'
        }

        self.logger.info(f"Parsers initialisés : {list(self.parsers.keys())}")

    def parse_file(self, file_path: str,
                   doc_type: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> ParsedDocument:
        """
        Parse un fichier unique.

        Args:
            file_path: Chemin vers le fichier
            doc_type: Type forcé (si None, détection auto)
            metadata: Métadonnées additionnelles

        Returns:
            ParsedDocument avec toutes les informations extraites

        Raises:
            ParsingError si le parsing échoue
        """
        file_path = Path(file_path)

        # Vérifications préliminaires
        if not file_path.exists():
            raise ParsingError(f"Le fichier n'existe pas : {file_path}")

        if not file_path.is_file():
            raise ParsingError(f"Le chemin ne pointe pas vers un fichier : {file_path}")

        # Vérifier la taille
        file_size = file_path.stat().st_size
        if file_size > self.file_size_limit:
            raise ParsingError(
                f"Fichier trop volumineux : {file_size / 1024 / 1024:.1f}MB "
                f"(limite : {self.file_size_limit / 1024 / 1024:.1f}MB)"
            )

        self.logger.info(f"Parsing du fichier : {file_path}")
        start_time = datetime.now()

        try:
            # Vérifier le cache
            if self.cache_enabled:
                cached_doc = self._get_from_cache(file_path)
                if cached_doc:
                    self.logger.info(f"Document trouvé dans le cache : {file_path}")
                    self.parsing_stats['cached'] += 1
                    return cached_doc

            # Déterminer le type de parser à utiliser
            if doc_type:
                parser_type = doc_type
            else:
                parser_type = self._detect_parser_type(file_path)

            if parser_type not in self.parsers:
                raise ParsingError(f"Type de parser non supporté : {parser_type}")

            # Sélectionner le parser
            parser = self.parsers[parser_type]

            # Ajouter les métadonnées système
            system_metadata = {
                'original_path': str(file_path),
                'file_name': file_path.name,
                'parser_type': parser_type,
                'parsing_timestamp': datetime.now().isoformat()
            }

            if metadata:
                system_metadata.update(metadata)

            # Parser le document
            parsed_doc = parser.parse(str(file_path))

            # Enrichir avec les métadonnées système
            parsed_doc.metadata.update(system_metadata)

            # Mettre en cache si activé
            if self.cache_enabled:
                self._save_to_cache(file_path, parsed_doc)

            # Mettre à jour les statistiques
            self._update_stats(parser_type, True, datetime.now() - start_time)

            self.logger.info(
                f"Parsing réussi : {file_path} "
                f"({len(parsed_doc.content)} caractères, "
                f"confiance : {parsed_doc.confidence_score:.2f})"
            )

            return parsed_doc

        except Exception as e:
            # Logger l'erreur
            error_info = {
                'file': str(file_path),
                'parser_type': parser_type if 'parser_type' in locals() else 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.error_log.append(error_info)

            # Mettre à jour les stats
            self._update_stats(
                parser_type if 'parser_type' in locals() else 'unknown',
                False,
                datetime.now() - start_time
            )

            self.logger.error(f"Échec du parsing de {file_path} : {str(e)}")
            raise ParsingError(f"Échec du parsing : {str(e)}")

    def parse_directory(self,
                        directory_path: str,
                        recursive: bool = True,
                        file_patterns: Optional[List[str]] = None,
                        exclude_patterns: Optional[List[str]] = None,
                        metadata_callback: Optional[callable] = None) -> List[ParsedDocument]:
        """
        Parse tous les documents d'un répertoire.

        Args:
            directory_path: Chemin vers le répertoire
            recursive: Parcourir les sous-répertoires
            file_patterns: Patterns de fichiers à inclure (ex: ['*.pdf', '*.md'])
            exclude_patterns: Patterns à exclure (ex: ['*_draft.*', 'temp/*'])
            metadata_callback: Fonction pour générer des métadonnées par fichier

        Returns:
            Liste des documents parsés avec succès
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Le répertoire n'existe pas : {directory}")

        if not directory.is_dir():
            raise ValueError(f"Le chemin n'est pas un répertoire : {directory}")

        self.logger.info(f"Parsing du répertoire : {directory} (récursif : {recursive})")

        # Collecter les fichiers à parser
        files_to_parse = self._collect_files(
            directory,
            recursive,
            file_patterns,
            exclude_patterns
        )

        self.logger.info(f"Fichiers à parser : {len(files_to_parse)}")

        # Parser les fichiers
        if self.parallel_processing and len(files_to_parse) > 1:
            parsed_documents = self._parse_files_parallel(
                files_to_parse,
                metadata_callback
            )
        else:
            parsed_documents = self._parse_files_sequential(
                files_to_parse,
                metadata_callback
            )

        # Générer le rapport
        self._generate_parsing_report(parsed_documents)

        return parsed_documents

    def _detect_parser_type(self, file_path: Path) -> str:
        """
        Détecte automatiquement le type de parser à utiliser.

        Cette méthode utilise plusieurs stratégies :
        1. Extension du fichier
        2. Type MIME
        3. Analyse du contenu (pour les cas ambigus)
        """
        # Stratégie 1 : Extension
        extension = file_path.suffix.lower()
        if extension in self.extension_mapping:
            parser_type = self.extension_mapping[extension]

            # Pour les types ambigus, faire une vérification supplémentaire
            if parser_type == 'faq' and extension in ['.txt', '.html', '.htm']:
                # Vérifier si c'est vraiment une FAQ
                if self.parsers['faq'].can_parse(str(file_path)):
                    return 'faq'
                else:
                    # Essayer d'autres parsers
                    for ptype, parser in self.parsers.items():
                        if ptype != 'faq' and parser.can_parse(str(file_path)):
                            return ptype

            return parser_type

        # Stratégie 2 : Type MIME
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in self.mime_mapping:
            return self.mime_mapping[mime_type]

        # Stratégie 3 : Demander à chaque parser s'il peut traiter le fichier
        for parser_type, parser in self.parsers.items():
            if parser.can_parse(str(file_path)):
                return parser_type

        # Si aucun parser ne reconnaît le fichier
        raise ParsingError(
            f"Impossible de déterminer le type de fichier : {file_path}\n"
            f"Extension : {extension}, MIME : {mime_type}"
        )

    def _collect_files(self,
                       directory: Path,
                       recursive: bool,
                       file_patterns: Optional[List[str]],
                       exclude_patterns: Optional[List[str]]) -> List[Path]:
        """
        Collecte les fichiers à parser selon les critères.
        """
        files = []

        # Si pas de patterns spécifiés, prendre toutes les extensions supportées
        if not file_patterns:
            file_patterns = [f"*{ext}" for ext in self.extension_mapping.keys()]

        # Fonction pour vérifier les exclusions
        def should_exclude(file_path: Path) -> bool:
            if not exclude_patterns:
                return False

            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    return True
            return False

        # Collecter les fichiers
        for pattern in file_patterns:
            if recursive:
                matches = directory.rglob(pattern)
            else:
                matches = directory.glob(pattern)

            for file_path in matches:
                if file_path.is_file() and not should_exclude(file_path):
                    files.append(file_path)

        # Éliminer les doublons et trier
        files = sorted(list(set(files)))

        return files

    def _parse_files_sequential(self,
                                files: List[Path],
                                metadata_callback: Optional[callable]) -> List[ParsedDocument]:
        """
        Parse les fichiers de manière séquentielle.
        """
        parsed_documents = []

        for i, file_path in enumerate(files, 1):
            self.logger.info(f"Parsing {i}/{len(files)} : {file_path.name}")

            try:
                # Générer les métadonnées si callback fourni
                metadata = None
                if metadata_callback:
                    metadata = metadata_callback(file_path)

                # Parser le fichier
                parsed_doc = self.parse_file(str(file_path), metadata=metadata)
                parsed_documents.append(parsed_doc)

            except Exception as e:
                self.logger.error(f"Erreur lors du parsing de {file_path} : {e}")
                continue

        return parsed_documents

    def _parse_files_parallel(self,
                              files: List[Path],
                              metadata_callback: Optional[callable]) -> List[ParsedDocument]:
        """
        Parse les fichiers en parallèle pour améliorer les performances.

        Utilise un pool de threads pour parser plusieurs fichiers simultanément.
        """
        parsed_documents = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumettre tous les fichiers
            future_to_file = {}

            for file_path in files:
                # Générer les métadonnées
                metadata = None
                if metadata_callback:
                    metadata = metadata_callback(file_path)

                # Soumettre la tâche
                future = executor.submit(
                    self.parse_file,
                    str(file_path),
                    metadata=metadata
                )
                future_to_file[future] = file_path

            # Collecter les résultats
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    parsed_doc = future.result()
                    parsed_documents.append(parsed_doc)
                    self.logger.info(f"✓ Parsé avec succès : {file_path.name}")

                except Exception as e:
                    self.logger.error(f"✗ Échec du parsing de {file_path} : {e}")

        return parsed_documents

    def _get_from_cache(self, file_path: Path) -> Optional[ParsedDocument]:
        """
        Récupère un document du cache s'il existe et est valide.
        """
        # Calculer le hash du fichier
        file_hash = self._calculate_file_hash(file_path)
        cache_file = self.cache_dir / f"{file_hash}.json"

        if not cache_file.exists():
            return None

        try:
            # Vérifier si le cache est toujours valide
            file_mtime = file_path.stat().st_mtime
            cache_mtime = cache_file.stat().st_mtime

            if file_mtime > cache_mtime:
                # Le fichier a été modifié après la mise en cache
                self.logger.debug(f"Cache invalidé pour {file_path} (fichier modifié)")
                cache_file.unlink()  # Supprimer le cache invalide
                return None

            # Charger depuis le cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Reconstruire le ParsedDocument
            parsed_doc = ParsedDocument(
                content=cache_data['content'],
                metadata=cache_data['metadata'],
                source_path=cache_data['source_path'],
                doc_type=cache_data['doc_type'],
                sections=cache_data['sections'],
                tables=cache_data['tables'],
                images_count=cache_data['images_count'],
                parsing_errors=cache_data['parsing_errors'],
                parsing_timestamp=datetime.fromisoformat(cache_data['parsing_timestamp']),
                document_hash=cache_data['document_hash'],
                confidence_score=cache_data['confidence_score']
            )

            return parsed_doc

        except Exception as e:
            self.logger.warning(f"Erreur lors de la lecture du cache : {e}")
            # Supprimer le cache corrompu
            if cache_file.exists():
                cache_file.unlink()
            return None

    def _save_to_cache(self, file_path: Path, parsed_doc: ParsedDocument):
        """
        Sauvegarde un document parsé dans le cache.
        """
        try:
            # Calculer le hash du fichier
            file_hash = self._calculate_file_hash(file_path)
            cache_file = self.cache_dir / f"{file_hash}.json"

            # Convertir en dictionnaire sérialisable
            cache_data = {
                'content': parsed_doc.content,
                'metadata': parsed_doc.metadata,
                'source_path': parsed_doc.source_path,
                'doc_type': parsed_doc.doc_type,
                'sections': parsed_doc.sections,
                'tables': parsed_doc.tables,
                'images_count': parsed_doc.images_count,
                'parsing_errors': parsed_doc.parsing_errors,
                'parsing_timestamp': parsed_doc.parsing_timestamp.isoformat(),
                'document_hash': parsed_doc.document_hash,
                'confidence_score': parsed_doc.confidence_score
            }

            # Sauvegarder
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.warning(f"Erreur lors de la sauvegarde en cache : {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calcule un hash unique pour un fichier.

        Utilise le chemin et la taille pour éviter de lire tout le fichier.
        """
        # Combiner le chemin absolu et la taille
        unique_string = f"{file_path.absolute()}_{file_path.stat().st_size}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _update_stats(self, parser_type: str, success: bool, duration):
        """
        Met à jour les statistiques de parsing.
        """
        self.parsing_stats['total_processed'] += 1

        if success:
            self.parsing_stats['successful'] += 1
        else:
            self.parsing_stats['failed'] += 1

        # Stats par type
        if parser_type not in self.parsing_stats['by_type']:
            self.parsing_stats['by_type'][parser_type] = {
                'total': 0,
                'successful': 0,
                'failed': 0
            }

        self.parsing_stats['by_type'][parser_type]['total'] += 1
        if success:
            self.parsing_stats['by_type'][parser_type]['successful'] += 1
        else:
            self.parsing_stats['by_type'][parser_type]['failed'] += 1

        # Temps total
        self.parsing_stats['total_time'] += duration.total_seconds()

    def _generate_parsing_report(self, parsed_documents: List[ParsedDocument]):
        """
        Génère un rapport détaillé du parsing.
        """
        report_lines = [
            "=" * 60,
            "RAPPORT DE PARSING",
            "=" * 60,
            f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RÉSUMÉ",
            "-" * 30,
            f"Documents traités : {self.parsing_stats['total_processed']}",
            f"Réussis : {self.parsing_stats['successful']}",
            f"Échecs : {self.parsing_stats['failed']}",
            f"Depuis le cache : {self.parsing_stats['cached']}",
            f"Temps total : {self.parsing_stats['total_time']:.2f}s",
            ""
        ]

        # Détails par type
        if self.parsing_stats['by_type']:
            report_lines.extend([
                "PAR TYPE DE DOCUMENT",
                "-" * 30
            ])

            for ptype, stats in self.parsing_stats['by_type'].items():
                success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
                report_lines.append(
                    f"{ptype.upper()} : {stats['total']} documents "
                    f"({success_rate:.1f}% de succès)"
                )

        # Documents parsés avec succès
        if parsed_documents:
            report_lines.extend([
                "",
                "DOCUMENTS PARSÉS AVEC SUCCÈS",
                "-" * 30
            ])

            for doc in parsed_documents[:10]:  # Limiter à 10 pour la lisibilité
                report_lines.append(
                    f"- {doc.metadata.get('file_name', 'Unknown')} "
                    f"({doc.doc_type}, {len(doc.content)} chars, "
                    f"confiance: {doc.confidence_score:.2f})"
                )

            if len(parsed_documents) > 10:
                report_lines.append(f"... et {len(parsed_documents) - 10} autres")

        # Erreurs
        if self.error_log:
            report_lines.extend([
                "",
                "ERREURS RENCONTRÉES",
                "-" * 30
            ])

            for error in self.error_log[-5:]:  # Dernières 5 erreurs
                report_lines.append(
                    f"- {error['file']} : {error['error'][:100]}..."
                )

        # Afficher le rapport
        report = "\n".join(report_lines)
        self.logger.info("\n" + report)

        # Sauvegarder le rapport si demandé
        if self.config.get('save_reports', True):
            report_file = self.cache_dir / f"parsing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

    def get_stats(self) -> Dict:
        """
        Retourne les statistiques de parsing actuelles.
        """
        return self.parsing_stats.copy()

    def clear_cache(self):
        """
        Vide le cache des documents parsés.
        """
        if not self.cache_enabled:
            return

        cache_files = list(self.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            cache_file.unlink()

        self.logger.info(f"Cache vidé : {len(cache_files)} fichiers supprimés")

    def get_supported_extensions(self) -> List[str]:
        """
        Retourne la liste des extensions supportées.
        """
        return list(self.extension_mapping.keys())

    def get_error_log(self) -> List[Dict]:
        """
        Retourne le journal des erreurs.
        """
        return self.error_log.copy()
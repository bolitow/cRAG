# src/ingestion/parsers/pdf_parser.py
import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Imports pour le traitement PDF
try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .base_parser import BaseParser, ParsedDocument, ParsingError


class PDFParser(BaseParser):
    """
    Parser spécialisé pour les documents PDF.

    Ce parser est conçu pour extraire intelligemment le contenu des PDF,
    en préservant la structure et en gérant les cas complexes comme :
    - Les tableaux
    - Les documents scannés (OCR)
    - Les PDF protégés par mot de passe
    - Les documents multi-colonnes
    - Les en-têtes et pieds de page répétitifs
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le parser PDF avec sa configuration.

        Config options:
            - ocr_enabled: Activer l'OCR pour les PDF scannés
            - password: Mot de passe pour les PDF protégés
            - remove_headers: Retirer les en-têtes/pieds de page
            - extract_tables: Extraire les tableaux séparément
            - page_range: Tuple (start, end) pour limiter l'extraction
        """
        super().__init__(config)
        self.supported_extensions = ['.pdf']

        # Vérifier les bibliothèques disponibles
        self.pdf_library = self._select_pdf_library()
        if not self.pdf_library:
            raise ImportError(
                "Aucune bibliothèque PDF trouvée. Installez pdfplumber, PyPDF2 ou PyMuPDF:\n"
                "pip install pdfplumber PyPDF2 pymupdf"
            )

        # Configuration spécifique PDF
        self.ocr_enabled = config.get('ocr_enabled', False) if config else False
        self.password = config.get('password') if config else None
        self.remove_headers = config.get('remove_headers', True) if config else True
        self.extract_tables = config.get('extract_tables', True) if config else True
        self.page_range = config.get('page_range') if config else None

        # Logger pour le débogage
        self.logger = logging.getLogger(__name__)

    def _select_pdf_library(self) -> str:
        """
        Sélectionne la meilleure bibliothèque PDF disponible.

        Ordre de préférence :
        1. pdfplumber : Meilleur pour l'extraction de tableaux
        2. PyMuPDF : Plus rapide et robuste
        3. PyPDF2 : Fallback basique
        """
        if PDFPLUMBER_AVAILABLE:
            return "pdfplumber"
        elif PYMUPDF_AVAILABLE:
            return "pymupdf"
        elif PYPDF2_AVAILABLE:
            return "pypdf2"
        else:
            return None

    def can_parse(self, file_path: str) -> bool:
        """
        Vérifie si le fichier est un PDF valide.
        """
        if not os.path.exists(file_path):
            return False

        # Vérifier l'extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            return False

        # Vérifier la signature du fichier (magic bytes)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                return header == b'%PDF-'
        except:
            return False

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse le document PDF et extrait toutes les informations.

        Cette méthode orchestre tout le processus d'extraction en utilisant
        la bibliothèque PDF appropriée et en appliquant les traitements nécessaires.
        """
        if not self.can_parse(file_path):
            raise ParsingError(f"Le fichier {file_path} n'est pas un PDF valide")

        self.logger.info(f"Parsing du PDF : {file_path} avec {self.pdf_library}")

        # Initialiser les variables
        content = ""
        tables = []
        metadata = self.extract_metadata(file_path)
        parsing_errors = []
        sections = []
        images_count = 0

        try:
            # Extraire le contenu selon la bibliothèque disponible
            if self.pdf_library == "pdfplumber":
                content, tables, pdf_metadata, images_count = self._parse_with_pdfplumber(file_path)
            elif self.pdf_library == "pymupdf":
                content, tables, pdf_metadata, images_count = self._parse_with_pymupdf(file_path)
            else:  # pypdf2
                content, tables, pdf_metadata, images_count = self._parse_with_pypdf2(file_path)

            # Fusionner les métadonnées
            metadata.update(pdf_metadata)

            # Post-traitement du contenu
            if self.remove_headers:
                content = self._remove_headers_footers(content)

            # Nettoyer le contenu
            content = self._clean_content(content)

            # Extraire les sections
            sections = self.extract_sections(content)

            # Identifier le type de document
            doc_type = self.identify_document_type(content, metadata)

            # Calculer le hash du document
            doc_hash = self.calculate_document_hash(content)

            # Calculer le score de confiance
            confidence = self._calculate_confidence_score(content, tables, parsing_errors)

        except Exception as e:
            self.logger.error(f"Erreur lors du parsing : {str(e)}")
            parsing_errors.append(f"Erreur critique : {str(e)}")
            content = ""
            confidence = 0.0
            doc_type = "unknown"
            doc_hash = ""

        # Créer le document parsé
        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            source_path=file_path,
            doc_type=doc_type,
            sections=sections,
            tables=tables,
            images_count=images_count,
            parsing_errors=parsing_errors,
            parsing_timestamp=datetime.now(),
            document_hash=doc_hash,
            confidence_score=confidence
        )

        # Valider et mettre à jour les stats
        is_valid, warnings = self.validate_parsing_result(parsed_doc)
        self.update_stats(is_valid, len(warnings))

        if warnings:
            self.logger.warning(f"Avertissements pour {file_path}: {warnings}")

        return parsed_doc

    def _parse_with_pdfplumber(self, file_path: str) -> Tuple[str, List[Dict], Dict, int]:
        """
        Parse le PDF avec pdfplumber (meilleur pour les tableaux).

        pdfplumber est excellent pour :
        - Extraire des tableaux complexes
        - Préserver la mise en page
        - Gérer les PDF avec des structures complexes
        """
        import pdfplumber

        content_parts = []
        tables = []
        metadata = {}
        images_count = 0

        with pdfplumber.open(file_path, password=self.password) as pdf:
            # Extraire les métadonnées
            if pdf.metadata:
                metadata = {
                    "title": pdf.metadata.get('Title', ''),
                    "author": pdf.metadata.get('Author', ''),
                    "subject": pdf.metadata.get('Subject', ''),
                    "creator": pdf.metadata.get('Creator', ''),
                    "producer": pdf.metadata.get('Producer', ''),
                    "creation_date": str(pdf.metadata.get('CreationDate', '')),
                    "modification_date": str(pdf.metadata.get('ModDate', '')),
                    "pages_count": len(pdf.pages)
                }

            # Déterminer les pages à traiter
            start_page = 0
            end_page = len(pdf.pages)
            if self.page_range:
                start_page = max(0, self.page_range[0] - 1)
                end_page = min(len(pdf.pages), self.page_range[1])

            # Parcourir les pages
            for page_num in range(start_page, end_page):
                page = pdf.pages[page_num]

                # Extraire le texte
                page_text = page.extract_text()
                if page_text:
                    content_parts.append(f"[Page {page_num + 1}]\n{page_text}")

                # Extraire les tableaux si demandé
                if self.extract_tables:
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:  # Vérifier que le tableau n'est pas vide
                            tables.append({
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "data": table,
                                "rows": len(table),
                                "cols": len(table[0]) if table else 0,
                                "text_representation": self._table_to_text(table)
                            })

                # Compter les images
                if hasattr(page, 'images'):
                    images_count += len(page.images)

        content = "\n\n".join(content_parts)
        return content, tables, metadata, images_count

    def _parse_with_pymupdf(self, file_path: str) -> Tuple[str, List[Dict], Dict, int]:
        """
        Parse le PDF avec PyMuPDF (plus rapide et robuste).

        PyMuPDF est excellent pour :
        - La vitesse de traitement
        - L'extraction de texte de haute qualité
        - La gestion des PDF complexes ou corrompus
        """
        import fitz

        content_parts = []
        tables = []
        metadata = {}
        images_count = 0

        # Ouvrir le document
        doc = fitz.open(file_path)

        # Gérer le mot de passe si nécessaire
        if doc.is_encrypted and self.password:
            if not doc.authenticate(self.password):
                raise ParsingError("Mot de passe incorrect pour le PDF")

        # Extraire les métadonnées
        metadata = {
            "title": doc.metadata.get('title', ''),
            "author": doc.metadata.get('author', ''),
            "subject": doc.metadata.get('subject', ''),
            "keywords": doc.metadata.get('keywords', ''),
            "creator": doc.metadata.get('creator', ''),
            "producer": doc.metadata.get('producer', ''),
            "creation_date": doc.metadata.get('creationDate', ''),
            "modification_date": doc.metadata.get('modDate', ''),
            "pages_count": doc.page_count
        }

        # Déterminer les pages à traiter
        start_page = 0
        end_page = doc.page_count
        if self.page_range:
            start_page = max(0, self.page_range[0] - 1)
            end_page = min(doc.page_count, self.page_range[1])

        # Parcourir les pages
        for page_num in range(start_page, end_page):
            page = doc[page_num]

            # Extraire le texte avec préservation de la mise en page
            page_text = page.get_text("text")
            if page_text:
                content_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            # Extraire les tableaux (méthode basique)
            if self.extract_tables:
                # PyMuPDF n'a pas d'extraction native de tableaux
                # On peut détecter les structures tabulaires par analyse
                table_areas = self._detect_table_areas(page_text)
                for table_area in table_areas:
                    tables.append({
                        "page": page_num + 1,
                        "table_index": len(tables),
                        "data": table_area,
                        "text_representation": table_area
                    })

            # Compter les images
            image_list = page.get_images()
            images_count += len(image_list)

        doc.close()
        content = "\n\n".join(content_parts)
        return content, tables, metadata, images_count

    def _parse_with_pypdf2(self, file_path: str) -> Tuple[str, List[Dict], Dict, int]:
        """
        Parse le PDF avec PyPDF2 (fallback basique).

        PyPDF2 est le plus basique mais fonctionne dans la plupart des cas simples.
        """
        import PyPDF2

        content_parts = []
        tables = []  # PyPDF2 ne supporte pas l'extraction de tableaux
        metadata = {}
        images_count = 0

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Gérer le mot de passe
            if pdf_reader.is_encrypted:
                if self.password:
                    if not pdf_reader.decrypt(self.password):
                        raise ParsingError("Mot de passe incorrect pour le PDF")
                else:
                    raise ParsingError("Le PDF est protégé par mot de passe")

            # Extraire les métadonnées
            if pdf_reader.metadata:
                metadata = {
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "subject": pdf_reader.metadata.get('/Subject', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                    "producer": pdf_reader.metadata.get('/Producer', ''),
                    "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                    "modification_date": str(pdf_reader.metadata.get('/ModDate', '')),
                    "pages_count": len(pdf_reader.pages)
                }

            # Déterminer les pages à traiter
            start_page = 0
            end_page = len(pdf_reader.pages)
            if self.page_range:
                start_page = max(0, self.page_range[0] - 1)
                end_page = min(len(pdf_reader.pages), self.page_range[1])

            # Extraire le texte de chaque page
            for page_num in range(start_page, end_page):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    content_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        content = "\n\n".join(content_parts)
        return content, tables, metadata, images_count

    def _remove_headers_footers(self, content: str) -> str:
        """
        Retire les en-têtes et pieds de page répétitifs.

        Cette méthode détecte les lignes qui se répètent sur plusieurs pages
        (typiquement les en-têtes et pieds de page) et les supprime.
        """
        lines = content.split('\n')
        line_counts = {}

        # Compter les occurrences de chaque ligne
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) < 100:  # Ignorer les longues lignes
                line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1

        # Identifier les lignes répétitives (apparaissent sur > 30% des pages)
        total_pages = content.count('[Page ')
        repetitive_lines = set()
        for line, count in line_counts.items():
            if count > max(2, total_pages * 0.3):
                repetitive_lines.add(line)

        # Filtrer les lignes
        filtered_lines = []
        for line in lines:
            if line.strip() not in repetitive_lines:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _clean_content(self, content: str) -> str:
        """
        Nettoie le contenu extrait des artifacts indésirables.

        Cette méthode supprime :
        - Les espaces multiples
        - Les caractères de contrôle
        - Les sauts de ligne excessifs
        - Les caractères non imprimables
        """
        # Remplacer les caractères de contrôle
        content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)

        # Normaliser les espaces
        content = re.sub(r' +', ' ', content)

        # Normaliser les sauts de ligne
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Retirer les marqueurs de page si demandé
        if self.config.get('remove_page_markers', False):
            content = re.sub(r'\[Page \d+\]\n?', '', content)

        # Nettoyer les tirets de césure
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

        return content.strip()

    def _table_to_text(self, table: List[List]) -> str:
        """
        Convertit un tableau en représentation textuelle lisible.

        Cette méthode transforme les données tabulaires en texte structuré
        qui peut être compris par le système de knowledge stripping.
        """
        if not table:
            return ""

        text_parts = []

        # Première ligne comme en-têtes
        if len(table) > 0:
            headers = table[0]
            text_parts.append("Tableau avec colonnes : " + " | ".join(str(h) for h in headers))

        # Lignes de données
        for row in table[1:]:
            row_text = " | ".join(str(cell) if cell else "-" for cell in row)
            text_parts.append(row_text)

        return "\n".join(text_parts)

    def _detect_table_areas(self, text: str) -> List[str]:
        """
        Détecte les zones de tableau dans le texte (heuristique simple).

        Cette méthode cherche des patterns qui ressemblent à des tableaux :
        - Alignement de | ou de tabs
        - Lignes avec un nombre régulier de séparateurs
        """
        table_areas = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Détecter le début d'un tableau
            if '|' in line or '\t' in line or re.search(r'\s{2,}', line):
                table_lines = [line]
                j = i + 1

                # Collecter les lignes suivantes qui font partie du tableau
                while j < len(lines):
                    next_line = lines[j]
                    if ('|' in next_line or '\t' in next_line or
                            re.search(r'\s{2,}', next_line) or
                            re.match(r'^[-=+]+$', next_line.strip())):
                        table_lines.append(next_line)
                        j += 1
                    else:
                        break

                # Si on a trouvé au moins 2 lignes, c'est probablement un tableau
                if len(table_lines) >= 2:
                    table_areas.append('\n'.join(table_lines))
                    i = j
                    continue

            i += 1

        return table_areas

    def _calculate_confidence_score(self, content: str, tables: List[Dict], errors: List[str]) -> float:
        """
        Calcule un score de confiance pour la qualité du parsing.

        Le score est basé sur :
        - La quantité de contenu extrait
        - La présence d'erreurs
        - La cohérence du texte
        - La détection réussie de structures (tableaux, sections)
        """
        score = 1.0

        # Pénalités pour les erreurs
        score -= len(errors) * 0.1

        # Pénalité si peu de contenu
        if len(content) < 100:
            score -= 0.3
        elif len(content) < 500:
            score -= 0.1

        # Bonus si des tableaux ont été détectés
        if tables:
            score += 0.1

        # Vérifier la cohérence du texte
        # Ratio de caractères alphabétiques
        alpha_chars = sum(1 for c in content if c.isalpha())
        if len(content) > 0:
            alpha_ratio = alpha_chars / len(content)
            if alpha_ratio < 0.5:
                score -= 0.2  # Probablement du texte corrompu

        # Vérifier la présence de mots français/anglais communs
        common_words = ['le', 'la', 'les', 'de', 'et', 'est', 'the', 'and', 'is', 'of']
        content_lower = content.lower()
        common_found = sum(1 for word in common_words if word in content_lower)
        if common_found < 3:
            score -= 0.1  # Peut-être un problème de langue ou d'extraction

        return max(0.0, min(1.0, score))
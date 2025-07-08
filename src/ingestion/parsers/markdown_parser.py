# src/ingestion/parsers/markdown_parser.py
import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Import pour le traitement Markdown avancé
try:
    import markdown
    from markdown.extensions import tables, fenced_code, footnotes, toc

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from .base_parser import BaseParser, ParsedDocument, ParsingError


class MarkdownParser(BaseParser):
    """
    Parser spécialisé pour les documents Markdown.

    Ce parser comprend la structure hiérarchique naturelle du Markdown
    et préserve les éléments importants comme :
    - La hiérarchie des titres (# ## ###)
    - Les listes ordonnées et non-ordonnées
    - Les tableaux
    - Les blocs de code
    - Les liens et références
    - Les métadonnées front matter (YAML)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le parser Markdown.

        Config options:
            - extract_front_matter: Extraire les métadonnées YAML
            - preserve_formatting: Garder les éléments de formatage
            - extract_code_blocks: Extraire les blocs de code séparément
            - extract_links: Collecter tous les liens du document
        """
        super().__init__(config)
        self.supported_extensions = ['.md', '.markdown', '.mdown', '.mkd']

        # Configuration spécifique Markdown
        self.extract_front_matter = config.get('extract_front_matter', True) if config else True
        self.preserve_formatting = config.get('preserve_formatting', False) if config else False
        self.extract_code_blocks = config.get('extract_code_blocks', True) if config else True
        self.extract_links = config.get('extract_links', True) if config else True

        # Logger
        self.logger = logging.getLogger(__name__)

        # Compiler les regex une seule fois pour la performance
        self._compile_patterns()

    def _compile_patterns(self):
        """
        Compile les patterns regex utilisés fréquemment.
        """
        # Pattern pour le front matter YAML
        self.front_matter_pattern = re.compile(
            r'^---\s*\n(.*?)\n---\s*\n',
            re.DOTALL | re.MULTILINE
        )

        # Pattern pour les headers
        self.header_pattern = re.compile(
            r'^(#{1,6})\s+(.+)$',
            re.MULTILINE
        )

        # Pattern pour les listes
        self.list_pattern = re.compile(
            r'^(\s*)([-*+]|\d+\.)\s+(.+)$',
            re.MULTILINE
        )

        # Pattern pour les tableaux
        self.table_pattern = re.compile(
            r'^\|.*\|$',
            re.MULTILINE
        )

        # Pattern pour les blocs de code
        self.code_block_pattern = re.compile(
            r'```(\w*)\n(.*?)\n```',
            re.DOTALL
        )

        # Pattern pour les liens
        self.link_pattern = re.compile(
            r'\[([^\]]+)\]\(([^)]+)\)'
        )

        # Pattern pour les images
        self.image_pattern = re.compile(
            r'!\[([^\]]*)\]\(([^)]+)\)'
        )

    def can_parse(self, file_path: str) -> bool:
        """
        Vérifie si le fichier est un document Markdown valide.
        """
        if not os.path.exists(file_path):
            return False

        # Vérifier l'extension
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse le document Markdown et extrait toutes les informations structurées.

        Cette méthode préserve la structure hiérarchique naturelle du Markdown
        tout en extrayant les éléments spéciaux pour un traitement ultérieur.
        """
        if not self.can_parse(file_path):
            raise ParsingError(f"Le fichier {file_path} n'est pas un fichier Markdown valide")

        self.logger.info(f"Parsing du fichier Markdown : {file_path}")

        # Lire le contenu du fichier
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            # Essayer avec un autre encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_content = f.read()

        # Initialiser les variables
        metadata = self.extract_metadata(file_path)
        parsing_errors = []
        tables = []
        sections = []
        code_blocks = []
        links = []

        try:
            # Extraire le front matter s'il existe
            content, front_matter = self._extract_front_matter(raw_content)
            if front_matter:
                metadata.update(front_matter)

            # Extraire les blocs de code avant tout traitement
            if self.extract_code_blocks:
                content, code_blocks = self._extract_code_blocks(content)

            # Extraire les tableaux
            tables = self._extract_tables(content)

            # Extraire les liens si demandé
            if self.extract_links:
                links = self._extract_links(content)
                metadata['links'] = links

            # Convertir en HTML pour une analyse plus facile si la lib est disponible
            if MARKDOWN_AVAILABLE and not self.preserve_formatting:
                html_content = self._convert_to_html(content)
                # Extraire le texte pur du HTML
                content = self._html_to_text(html_content)

            # Extraire les sections basées sur les headers
            sections = self._extract_sections_from_markdown(raw_content)

            # Identifier le type de document
            doc_type = self.identify_document_type(content, metadata)

            # Calculer le hash
            doc_hash = self.calculate_document_hash(content)

            # Calculer le score de confiance
            confidence = self._calculate_confidence_score(
                content, sections, tables, code_blocks, parsing_errors
            )

        except Exception as e:
            self.logger.error(f"Erreur lors du parsing : {str(e)}")
            parsing_errors.append(f"Erreur : {str(e)}")
            content = raw_content  # Fallback sur le contenu brut
            confidence = 0.5
            doc_type = "markdown"
            doc_hash = self.calculate_document_hash(raw_content)

        # Ajouter les métadonnées spécifiques au Markdown
        metadata.update({
            'format': 'markdown',
            'has_front_matter': bool(front_matter),
            'headers_count': len(sections),
            'code_blocks_count': len(code_blocks),
            'tables_count': len(tables),
            'links_count': len(links)
        })

        # Créer le document parsé
        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            source_path=file_path,
            doc_type=doc_type,
            sections=sections,
            tables=tables,
            images_count=len(re.findall(self.image_pattern, raw_content)),
            parsing_errors=parsing_errors,
            parsing_timestamp=datetime.now(),
            document_hash=doc_hash,
            confidence_score=confidence
        )

        # Si on a extrait des blocs de code, les ajouter aux métadonnées
        if code_blocks:
            parsed_doc.metadata['code_blocks'] = code_blocks

        # Valider et mettre à jour les stats
        is_valid, warnings = self.validate_parsing_result(parsed_doc)
        self.update_stats(is_valid, len(warnings))

        return parsed_doc

    def _extract_front_matter(self, content: str) -> Tuple[str, Dict]:
        """
        Extrait les métadonnées YAML du front matter.

        Le front matter est souvent utilisé dans les documents techniques
        pour stocker des métadonnées comme l'auteur, la date, les tags, etc.
        """
        match = self.front_matter_pattern.match(content)

        if not match:
            return content, {}

        try:
            import yaml
            front_matter_text = match.group(1)
            front_matter_data = yaml.safe_load(front_matter_text)

            # Retirer le front matter du contenu
            content_without_fm = content[match.end():]

            return content_without_fm, front_matter_data or {}
        except ImportError:
            self.logger.warning("PyYAML non installé, impossible d'extraire le front matter")
            return content, {}
        except Exception as e:
            self.logger.warning(f"Erreur lors du parsing du front matter : {e}")
            return content, {}

    def _extract_code_blocks(self, content: str) -> Tuple[str, List[Dict]]:
        """
        Extrait les blocs de code et les remplace par des marqueurs.

        Les blocs de code peuvent contenir des exemples de configuration,
        des scripts, ou des commandes importantes pour la cybersécurité.
        """
        code_blocks = []

        def replace_code_block(match):
            language = match.group(1) or "text"
            code_content = match.group(2)

            code_blocks.append({
                "language": language,
                "content": code_content,
                "lines_count": len(code_content.split('\n'))
            })

            # Remplacer par un marqueur
            return f"\n[BLOC DE CODE {language.upper()} - {len(code_blocks)}]\n"

        # Remplacer tous les blocs de code
        content_with_markers = self.code_block_pattern.sub(replace_code_block, content)

        return content_with_markers, code_blocks

    def _extract_tables(self, content: str) -> List[Dict]:
        """
        Extrait les tableaux Markdown.

        Les tableaux contiennent souvent des informations structurées importantes
        comme des matrices de contrôle, des listes de configurations, etc.
        """
        tables = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Détecter le début d'un tableau
            if self.table_pattern.match(line):
                table_lines = [line]
                j = i + 1

                # Collecter toutes les lignes du tableau
                while j < len(lines) and self.table_pattern.match(lines[j].strip()):
                    table_lines.append(lines[j].strip())
                    j += 1

                # Parser le tableau si on a au moins 2 lignes (header + separator)
                if len(table_lines) >= 2:
                    table_data = self._parse_markdown_table(table_lines)
                    if table_data:
                        tables.append({
                            "start_line": i,
                            "end_line": j - 1,
                            "data": table_data,
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                            "text_representation": self._table_to_text(table_data)
                        })

                i = j
            else:
                i += 1

        return tables

    def _parse_markdown_table(self, lines: List[str]) -> List[List[str]]:
        """
        Parse un tableau Markdown en structure de données.
        """
        if len(lines) < 2:
            return []

        table_data = []

        for line in lines:
            # Ignorer les lignes de séparation (---|---|---)
            if re.match(r'^[\|\s\-:]+$', line):
                continue

            # Extraire les cellules
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:
                table_data.append(cells)

        return table_data

    def _extract_links(self, content: str) -> List[Dict]:
        """
        Extrait tous les liens du document.

        Les liens peuvent pointer vers des ressources importantes,
        des références normatives, ou des documents connexes.
        """
        links = []

        # Liens inline
        for match in self.link_pattern.finditer(content):
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "type": "inline"
            })

        # Liens de référence [text][ref]
        ref_link_pattern = re.compile(r'\[([^\]]+)\]\[([^\]]+)\]')
        ref_def_pattern = re.compile(r'^\[([^\]]+)\]:\s*(.+)$', re.MULTILINE)

        # D'abord, collecter les définitions de référence
        ref_definitions = {}
        for match in ref_def_pattern.finditer(content):
            ref_definitions[match.group(1)] = match.group(2)

        # Puis, résoudre les liens de référence
        for match in ref_link_pattern.finditer(content):
            ref_id = match.group(2)
            if ref_id in ref_definitions:
                links.append({
                    "text": match.group(1),
                    "url": ref_definitions[ref_id],
                    "type": "reference"
                })

        return links

    def _extract_sections_from_markdown(self, content: str) -> List[Dict]:
        """
        Extrait les sections basées sur la hiérarchie des headers Markdown.

        Cette méthode préserve la structure hiérarchique du document,
        ce qui est crucial pour comprendre l'organisation des politiques
        et procédures.
        """
        sections = []
        lines = content.split('\n')

        current_section = None
        section_stack = []  # Pour gérer la hiérarchie

        for i, line in enumerate(lines):
            header_match = self.header_pattern.match(line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Créer une nouvelle section
                new_section = {
                    "title": title,
                    "level": level,
                    "start_line": i,
                    "content": "",
                    "subsections": []
                }

                # Gérer la hiérarchie
                # Retirer les sections de niveau supérieur ou égal de la pile
                while section_stack and section_stack[-1]["level"] >= level:
                    section_stack.pop()

                # Si on a une section parent dans la pile
                if section_stack:
                    # Ajouter comme sous-section
                    section_stack[-1]["subsections"].append(new_section)
                else:
                    # Section de niveau racine
                    sections.append(new_section)

                # Ajouter à la pile
                section_stack.append(new_section)
                current_section = new_section

            elif current_section and line.strip():
                # Ajouter le contenu à la section courante
                current_section["content"] += line + "\n"

        return sections

    def _convert_to_html(self, content: str) -> str:
        """
        Convertit le Markdown en HTML pour faciliter l'extraction de texte.

        Cette conversion permet de gérer correctement les éléments complexes
        comme les listes imbriquées, les tableaux, etc.
        """
        if not MARKDOWN_AVAILABLE:
            return content

        # Extensions utiles pour les documents techniques
        extensions = [
            'tables',  # Support des tableaux
            'fenced_code',  # Blocs de code avec ```
            'footnotes',  # Notes de bas de page
            'toc',  # Table des matières
            'nl2br',  # Convertir les sauts de ligne
            'sane_lists'  # Meilleures listes
        ]

        try:
            md = markdown.Markdown(extensions=extensions)
            html = md.convert(content)
            return html
        except Exception as e:
            self.logger.warning(f"Erreur lors de la conversion Markdown : {e}")
            return content

    def _html_to_text(self, html: str) -> str:
        """
        Extrait le texte pur du HTML.

        Cette méthode préserve la structure tout en retirant les balises HTML.
        """
        # Méthode simple sans dépendance externe
        # Pour une meilleure extraction, on pourrait utiliser BeautifulSoup

        # Remplacer les balises de bloc par des sauts de ligne
        block_tags = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']
        for tag in block_tags:
            html = re.sub(f'</{tag}>', '\n', html, flags=re.IGNORECASE)
            html = re.sub(f'<{tag}[^>]*>', '', html, flags=re.IGNORECASE)

        # Remplacer les <br> par des sauts de ligne
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)

        # Retirer toutes les autres balises
        html = re.sub(r'<[^>]+>', '', html)

        # Décoder les entités HTML
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        html = html.replace('&nbsp;', ' ')

        # Nettoyer les espaces multiples et les lignes vides
        html = re.sub(r' +', ' ', html)
        html = re.sub(r'\n{3,}', '\n\n', html)

        return html.strip()

    def _table_to_text(self, table: List[List[str]]) -> str:
        """
        Convertit un tableau en texte structuré.

        Adapté aux besoins de la cybersécurité pour préserver
        les relations entre les colonnes (ex: contrôle -> description -> criticité).
        """
        if not table:
            return ""

        text_parts = []

        # Headers
        if len(table) > 0:
            headers = table[0]
            text_parts.append("Colonnes : " + " | ".join(headers))
            text_parts.append("-" * 50)

        # Données
        for i, row in enumerate(table[1:], 1):
            row_parts = []
            for j, cell in enumerate(row):
                if j < len(table[0]):  # S'assurer qu'on a le header
                    header = table[0][j]
                    row_parts.append(f"{header}: {cell}")
                else:
                    row_parts.append(f"Col{j + 1}: {cell}")

            text_parts.append(f"Ligne {i}: " + " | ".join(row_parts))

        return "\n".join(text_parts)

    def _calculate_confidence_score(self, content: str, sections: List[Dict],
                                    tables: List[Dict], code_blocks: List[Dict],
                                    errors: List[str]) -> float:
        """
        Calcule le score de confiance pour le parsing Markdown.

        Le score reflète la qualité de l'extraction et la richesse
        de la structure détectée.
        """
        score = 1.0

        # Pénalités pour les erreurs
        score -= len(errors) * 0.15

        # Bonus pour la structure détectée
        if sections:
            score += min(0.2, len(sections) * 0.02)  # Max +0.2

        if tables:
            score += 0.1

        if code_blocks:
            score += 0.05

        # Vérifier la qualité du contenu
        if len(content) < 100:
            score -= 0.3

        # Vérifier la présence de structure Markdown
        markdown_elements = 0
        if '#' in content:
            markdown_elements += 1
        if '```' in content or '    ' in content:  # Code
            markdown_elements += 1
        if '|' in content:  # Tables
            markdown_elements += 1
        if '[' in content and ']' in content:  # Links
            markdown_elements += 1

        if markdown_elements == 0:
            score -= 0.2  # Peut-être pas vraiment du Markdown

        return max(0.0, min(1.0, score))
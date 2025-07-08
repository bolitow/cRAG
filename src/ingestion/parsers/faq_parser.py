# src/ingestion/parsers/faq_parser.py
import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from bs4 import BeautifulSoup

from .base_parser import BaseParser, ParsedDocument, ParsingError


class FAQParser(BaseParser):
    """
    Parser spécialisé pour les documents FAQ (Foire Aux Questions).

    Ce parser est conçu pour extraire intelligemment les paires question-réponse
    de différents formats de FAQ couramment utilisés en entreprise :
    - FAQ en Markdown avec différents styles
    - FAQ en HTML (exports de wikis, SharePoint, etc.)
    - FAQ en JSON structuré
    - FAQ en texte avec formats variés
    - FAQ en format Word converti

    Le parser préserve le contexte et les relations entre questions connexes,
    ce qui est crucial pour répondre efficacement aux audits.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le parser FAQ.

        Config options:
            - extract_categories: Extraire les catégories/sections de FAQ
            - merge_multipart: Fusionner les réponses multi-parties
            - extract_metadata: Extraire les métadonnées (auteur, date, etc.)
            - confidence_threshold: Seuil de confiance pour la détection Q/R
            - language: Langue principale (fr/en) pour améliorer la détection
        """
        super().__init__(config)
        self.supported_extensions = [
            '.faq', '.txt', '.md', '.html', '.htm', '.json',
            '.docx'  # Si converti en texte
        ]

        # Configuration
        self.extract_categories = config.get('extract_categories', True) if config else True
        self.merge_multipart = config.get('merge_multipart', True) if config else True
        self.extract_metadata = config.get('extract_metadata', True) if config else True
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        self.language = config.get('language', 'fr') if config else 'fr'

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialiser les patterns de détection
        self._init_detection_patterns()

    def _init_detection_patterns(self):
        """
        Initialise les patterns pour détecter les questions et réponses.

        Ces patterns couvrent les formats FAQ les plus courants en entreprise,
        en français et en anglais.
        """
        # Patterns pour les questions
        if self.language == 'fr':
            self.question_starters = [
                r'^(?:Q|Question)\s*(?:\d+)?\.?\s*:?\s*',
                r'^(?:\d+\.?\s*)?(?:Qu(?:\')?|Comment|Pourquoi|Quand|Où|Est-ce|Peut-on|Dois-je|Faut-il)',
                r'^\?\s*',  # Questions commençant par ?
                r'^.*\?\s*$'  # Phrases se terminant par ?
            ]

            self.answer_starters = [
                r'^(?:R|A|Réponse|Answer)\s*(?:\d+)?\.?\s*:?\s*',
                r'^→\s*',  # Flèche
                r'^>\s*',  # Citation style
            ]
        else:  # English
            self.question_starters = [
                r'^(?:Q|Question)\s*(?:\d+)?\.?\s*:?\s*',
                r'^(?:\d+\.?\s*)?(?:What|How|Why|When|Where|Is|Can|Should|Does|Do)',
                r'^\?\s*',
                r'^.*\?\s*$'
            ]

            self.answer_starters = [
                r'^(?:A|Answer)\s*(?:\d+)?\.?\s*:?\s*',
                r'^→\s*',
                r'^>\s*',
            ]

        # Patterns pour les catégories/sections
        self.category_patterns = [
            r'^#{1,3}\s+(.+)$',  # Headers Markdown
            r'^(?:Catégorie|Category|Section|Thème|Topic)\s*:?\s*(.+)$',
            r'^([A-Z][A-Z\s]+):?\s*$',  # Titres en majuscules
            r'^\[(.+)\]\s*$',  # [Catégorie]
        ]

        # Patterns pour les structures FAQ communes
        self.faq_structures = {
            'numbered': {
                'question': r'^(\d+)\.\s*(.+\?)\s*$',
                'answer': r'^(?:(?:\d+\.)?\s*)?(.+)$'
            },
            'qa_prefixed': {
                'question': r'^Q\s*:?\s*(.+)$',
                'answer': r'^[AR]\s*:?\s*(.+)$'
            },
            'markdown': {
                'question': r'^###?\s+(.+\?)?\s*$',
                'answer': r'^(.+)$'
            },
            'definition_list': {
                'question': r'^(.+)\s*::\s*$',
                'answer': r'^\s+(.+)$'
            },
            'bullet_points': {
                'question': r'^[•\-\*]\s*(.+\?)\s*$',
                'answer': r'^\s+(.+)$'
            }
        }

        # Mots-clés indiquant une FAQ
        self.faq_indicators = [
            'faq', 'frequently asked questions', 'questions fréquentes',
            'questions/réponses', 'q&a', 'q/a', 'questions et réponses',
            'foire aux questions', 'help', 'aide'
        ]

        # Patterns pour extraire les métadonnées
        self.metadata_patterns = {
            'author': r'(?:Auteur|Author|Rédigé par|Written by)\s*:?\s*(.+)',
            'date': r'(?:Date|Mise à jour|Updated|Last modified)\s*:?\s*(.+)',
            'version': r'(?:Version|V)\s*:?\s*(\d+\.?\d*)',
            'tags': r'(?:Tags?|Mots-clés|Keywords)\s*:?\s*(.+)'
        }

    def can_parse(self, file_path: str) -> bool:
        """
        Vérifie si le fichier peut être une FAQ.

        Cette méthode est plus permissive que les autres parsers car
        les FAQ peuvent avoir des extensions variées.
        """
        if not os.path.exists(file_path):
            return False

        # Vérifier l'extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            return False

        # Vérifier le nom du fichier pour des indices
        filename = os.path.basename(file_path).lower()
        if any(indicator in filename for indicator in ['faq', 'qa', 'questions']):
            return True

        # Pour les fichiers texte/markdown, vérifier le contenu
        if ext in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_lines = f.read(1000).lower()
                    return any(indicator in first_lines for indicator in self.faq_indicators)
            except:
                pass

        return True  # Donner une chance aux autres fichiers

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse le document FAQ et extrait les paires question-réponse.

        Cette méthode identifie automatiquement le format de FAQ et
        applique la stratégie d'extraction appropriée.
        """
        if not self.can_parse(file_path):
            raise ParsingError(f"Le fichier {file_path} n'est pas reconnu comme FAQ")

        self.logger.info(f"Parsing du fichier FAQ : {file_path}")

        # Déterminer le type de fichier
        ext = os.path.splitext(file_path)[1].lower()

        # Initialiser les variables
        metadata = self.extract_metadata(file_path)
        parsing_errors = []

        try:
            # Parser selon le format
            if ext == '.json':
                qa_pairs, categories, file_metadata = self._parse_json_faq(file_path)
            elif ext in ['.html', '.htm']:
                qa_pairs, categories, file_metadata = self._parse_html_faq(file_path)
            else:
                # Texte, Markdown, ou autre
                qa_pairs, categories, file_metadata = self._parse_text_faq(file_path)

            # Fusionner les métadonnées
            metadata.update(file_metadata)

            # Valider et enrichir les Q/R
            validated_pairs = self._validate_and_enrich_qa_pairs(qa_pairs)

            # Organiser par catégories si trouvées
            if categories and self.extract_categories:
                sections = self._organize_by_categories(validated_pairs, categories)
            else:
                sections = self._create_flat_sections(validated_pairs)

            # Générer le contenu textuel structuré
            content = self._generate_structured_content(validated_pairs, categories)

            # Créer les tables (une par catégorie)
            tables = self._create_faq_tables(validated_pairs, categories)

            # Statistiques
            metadata.update({
                'format': 'faq',
                'total_questions': len(validated_pairs),
                'categories_count': len(categories),
                'avg_answer_length': sum(len(qa['answer'].split()) for qa in validated_pairs) / len(
                    validated_pairs) if validated_pairs else 0
            })

            # Calculer le hash et la confiance
            doc_hash = self.calculate_document_hash(content)
            confidence = self._calculate_confidence_score(validated_pairs, categories, parsing_errors)

        except Exception as e:
            self.logger.error(f"Erreur lors du parsing : {str(e)}")
            parsing_errors.append(f"Erreur : {str(e)}")
            content = ""
            sections = []
            tables = []
            confidence = 0.0
            doc_hash = ""

        # Créer le document parsé
        parsed_doc = ParsedDocument(
            content=content,
            metadata=metadata,
            source_path=file_path,
            doc_type="faq",
            sections=sections,
            tables=tables,
            images_count=0,
            parsing_errors=parsing_errors,
            parsing_timestamp=datetime.now(),
            document_hash=doc_hash,
            confidence_score=confidence
        )

        # Valider et mettre à jour les stats
        is_valid, warnings = self.validate_parsing_result(parsed_doc)
        self.update_stats(is_valid, len(warnings))

        return parsed_doc

    def _parse_json_faq(self, file_path: str) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse une FAQ au format JSON.

        Formats JSON supportés :
        - Liste simple de {question, answer}
        - Structure avec catégories
        - Format avec métadonnées
        """
        qa_pairs = []
        categories = []
        metadata = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Cas 1: Liste simple
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    qa_pairs.append({
                        'id': f"Q{idx + 1}",
                        'question': str(item.get('question', item.get('q', ''))),
                        'answer': str(item.get('answer', item.get('a', item.get('response', '')))),
                        'category': item.get('category', 'Général'),
                        'metadata': {k: v for k, v in item.items() if k not in ['question', 'answer', 'q', 'a']}
                    })

        # Cas 2: Structure avec métadonnées
        elif isinstance(data, dict):
            # Extraire les métadonnées
            if 'metadata' in data:
                metadata = data['metadata']

            # Extraire les FAQ
            faq_data = data.get('faq', data.get('questions', data))

            if isinstance(faq_data, list):
                # Traiter comme liste simple
                for idx, item in enumerate(faq_data):
                    qa_pairs.append({
                        'id': f"Q{idx + 1}",
                        'question': str(item.get('question', item.get('q', ''))),
                        'answer': str(item.get('answer', item.get('a', ''))),
                        'category': item.get('category', 'Général'),
                        'metadata': item.get('metadata', {})
                    })

            elif isinstance(faq_data, dict):
                # Structure par catégories
                for cat_name, cat_questions in faq_data.items():
                    categories.append({
                        'name': cat_name,
                        'description': '',
                        'question_count': len(cat_questions) if isinstance(cat_questions, list) else 0
                    })

                    if isinstance(cat_questions, list):
                        for idx, item in enumerate(cat_questions):
                            qa_pairs.append({
                                'id': f"{cat_name}-Q{idx + 1}",
                                'question': str(item.get('question', item.get('q', ''))),
                                'answer': str(item.get('answer', item.get('a', ''))),
                                'category': cat_name,
                                'metadata': item.get('metadata', {})
                            })

        return qa_pairs, categories, metadata

    def _parse_html_faq(self, file_path: str) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse une FAQ au format HTML.

        Gère les formats HTML courants :
        - Definition lists (dl/dt/dd)
        - Sections avec headers
        - Accordéons/toggles
        - Tables Q/R
        """
        qa_pairs = []
        categories = []
        metadata = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extraire les métadonnées du HTML
        if soup.find('meta', {'name': 'author'}):
            metadata['author'] = soup.find('meta', {'name': 'author'}).get('content', '')
        if soup.find('meta', {'name': 'description'}):
            metadata['description'] = soup.find('meta', {'name': 'description'}).get('content', '')

        # Méthode 1: Definition lists (dl/dt/dd)
        for dl in soup.find_all('dl'):
            category = self._find_previous_header(dl)
            questions = dl.find_all('dt')
            answers = dl.find_all('dd')

            for idx, (q, a) in enumerate(zip(questions, answers)):
                qa_pairs.append({
                    'id': f"DL{len(qa_pairs) + 1}",
                    'question': q.get_text(strip=True),
                    'answer': a.get_text(strip=True),
                    'category': category or 'Général',
                    'metadata': {}
                })

        # Méthode 2: Sections avec classes FAQ
        faq_containers = soup.find_all(['div', 'section'], class_=re.compile(r'faq|qa|question'))
        for container in faq_containers:
            # Chercher les paires Q/R dans le container
            questions = container.find_all(['h3', 'h4', 'div'], class_=re.compile(r'question'))
            answers = container.find_all(['p', 'div'], class_=re.compile(r'answer|response'))

            for q, a in zip(questions, answers):
                qa_pairs.append({
                    'id': f"FAQ{len(qa_pairs) + 1}",
                    'question': q.get_text(strip=True),
                    'answer': a.get_text(strip=True),
                    'category': self._find_container_category(container) or 'Général',
                    'metadata': {}
                })

        # Méthode 3: Accordéons (details/summary)
        for details in soup.find_all('details'):
            summary = details.find('summary')
            if summary:
                # Le contenu après summary est la réponse
                answer_parts = []
                for elem in summary.find_next_siblings():
                    answer_parts.append(elem.get_text(strip=True))

                qa_pairs.append({
                    'id': f"ACC{len(qa_pairs) + 1}",
                    'question': summary.get_text(strip=True),
                    'answer': ' '.join(answer_parts),
                    'category': self._find_previous_header(details) or 'Général',
                    'metadata': {}
                })

        # Méthode 4: Pattern Q:/A: ou Question:/Answer:
        text_content = soup.get_text()
        qa_pattern = re.compile(
            r'(?:Q|Question)\s*:?\s*(.+?)(?:\n|\r\n)?\s*(?:A|Answer|R|Réponse)\s*:?\s*(.+?)(?=(?:Q|Question)\s*:|$)',
            re.IGNORECASE | re.DOTALL
        )

        for match in qa_pattern.finditer(text_content):
            qa_pairs.append({
                'id': f"TXT{len(qa_pairs) + 1}",
                'question': match.group(1).strip(),
                'answer': match.group(2).strip(),
                'category': 'Général',
                'metadata': {}
            })

        # Extraire les catégories des headers
        for header in soup.find_all(['h1', 'h2', 'h3']):
            header_text = header.get_text(strip=True)
            if not any(indicator in header_text.lower() for indicator in ['faq', 'questions']):
                categories.append({
                    'name': header_text,
                    'description': '',
                    'question_count': 0  # Sera calculé après
                })

        return qa_pairs, categories, metadata

    def _parse_text_faq(self, file_path: str) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Parse une FAQ en format texte ou Markdown.

        Cette méthode utilise une approche multi-passes pour détecter
        le format et extraire les Q/R.
        """
        qa_pairs = []
        categories = []
        metadata = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extraire les métadonnées du contenu
        metadata = self._extract_text_metadata(content)

        # Détecter le format principal
        detected_format = self._detect_faq_format(content)
        self.logger.info(f"Format FAQ détecté : {detected_format}")

        # Parser selon le format détecté
        if detected_format == 'numbered':
            qa_pairs = self._parse_numbered_faq(content)
        elif detected_format == 'qa_prefixed':
            qa_pairs = self._parse_qa_prefixed_faq(content)
        elif detected_format == 'markdown':
            qa_pairs = self._parse_markdown_faq(content)
        elif detected_format == 'definition_list':
            qa_pairs = self._parse_definition_list_faq(content)
        else:
            # Fallback: recherche générique de questions
            qa_pairs = self._parse_generic_faq(content)

        # Extraire les catégories
        categories = self._extract_categories_from_text(content)

        # Assigner les catégories aux questions
        qa_pairs = self._assign_categories_to_questions(qa_pairs, categories, content)

        return qa_pairs, categories, metadata

    def _detect_faq_format(self, content: str) -> str:
        """
        Détecte automatiquement le format de FAQ utilisé.

        Cette détection permet d'appliquer la meilleure stratégie de parsing.
        """
        lines = content.split('\n')
        format_scores = {fmt: 0 for fmt in self.faq_structures.keys()}

        # Tester chaque format sur un échantillon
        sample_size = min(50, len(lines))

        for i in range(sample_size):
            line = lines[i].strip()
            if not line:
                continue

            # Tester chaque format
            for fmt_name, patterns in self.faq_structures.items():
                if re.match(patterns['question'], line):
                    format_scores[fmt_name] += 2

                    # Vérifier si la ligne suivante correspond au pattern de réponse
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if re.match(patterns.get('answer', '.*'), next_line):
                            format_scores[fmt_name] += 3

        # Retourner le format avec le score le plus élevé
        best_format = max(format_scores, key=format_scores.get)

        # Si aucun format n'a un score significatif, utiliser generic
        if format_scores[best_format] < 5:
            return 'generic'

        return best_format

    def _parse_numbered_faq(self, content: str) -> List[Dict]:
        """
        Parse une FAQ avec questions numérotées.

        Format:
        1. Question?
        Réponse...

        2. Question?
        Réponse...
        """
        qa_pairs = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Chercher une question numérotée
            match = re.match(r'^(\d+)\.\s*(.+\?)\s*$', line)
            if match:
                q_num = match.group(1)
                question = match.group(2)

                # Collecter la réponse (lignes suivantes jusqu'à la prochaine question)
                answer_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Si on trouve la prochaine question numérotée, arrêter
                    if re.match(r'^\d+\.\s*.+\?', next_line):
                        break

                    if next_line:  # Ignorer les lignes vides
                        answer_lines.append(next_line)
                    elif answer_lines:  # Ligne vide après du contenu = fin de réponse
                        break

                    j += 1

                if answer_lines:
                    qa_pairs.append({
                        'id': f"Q{q_num}",
                        'question': question,
                        'answer': ' '.join(answer_lines),
                        'category': 'Général',
                        'metadata': {'format': 'numbered'}
                    })

                i = j
            else:
                i += 1

        return qa_pairs

    def _parse_qa_prefixed_faq(self, content: str) -> List[Dict]:
        """
        Parse une FAQ avec préfixes Q:/A: ou Question:/Answer:.
        """
        qa_pairs = []

        # Pattern pour capturer Q/A avec variations
        patterns = [
            # Q: ... A: ...
            re.compile(
                r'(?:Q|Question)\s*(?:\d+)?\s*:\s*(.+?)(?:\n|\r\n)?\s*(?:A|Answer|R|Réponse)\s*:\s*(.+?)(?=(?:Q|Question)\s*(?:\d+)?\s*:|$)',
                re.IGNORECASE | re.DOTALL
            ),
            # Q. ... A. ...
            re.compile(
                r'(?:Q|Question)\s*(?:\d+)?\s*\.\s*(.+?)(?:\n|\r\n)?\s*(?:A|Answer|R|Réponse)\s*\.\s*(.+?)(?=(?:Q|Question)\s*(?:\d+)?\s*\.|$)',
                re.IGNORECASE | re.DOTALL
            ),
        ]

        for pattern in patterns:
            matches = list(pattern.finditer(content))
            if matches:
                for idx, match in enumerate(matches):
                    qa_pairs.append({
                        'id': f"QA{idx + 1}",
                        'question': match.group(1).strip(),
                        'answer': match.group(2).strip(),
                        'category': 'Général',
                        'metadata': {'format': 'qa_prefixed'}
                    })
                break  # Utiliser le premier pattern qui fonctionne

        return qa_pairs

    def _parse_markdown_faq(self, content: str) -> List[Dict]:
        """
        Parse une FAQ au format Markdown avec headers pour questions.
        """
        qa_pairs = []
        lines = content.split('\n')

        current_category = 'Général'
        i = 0

        while i < len(lines):
            line = lines[i]

            # Détecter les catégories (## Header)
            if re.match(r'^##\s+(.+)$', line):
                match = re.match(r'^##\s+(.+)$', line)
                if match and '?' not in match.group(1):
                    current_category = match.group(1).strip()

            # Détecter les questions (### Question?)
            elif re.match(r'^###\s+(.+\?)$', line):
                match = re.match(r'^###\s+(.+\?)$', line)
                if match:
                    question = match.group(1).strip()

                    # Collecter la réponse
                    answer_lines = []
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]

                        # Arrêter à la prochaine question ou catégorie
                        if re.match(r'^##', next_line):
                            break

                        if next_line.strip():
                            answer_lines.append(next_line.strip())

                        j += 1

                    if answer_lines:
                        qa_pairs.append({
                            'id': f"MD{len(qa_pairs) + 1}",
                            'question': question,
                            'answer': ' '.join(answer_lines),
                            'category': current_category,
                            'metadata': {'format': 'markdown'}
                        })

                    i = j - 1

            i += 1

        return qa_pairs

    def _parse_definition_list_faq(self, content: str) -> List[Dict]:
        """
        Parse une FAQ au format liste de définitions.

        Format:
        Question::
            Réponse indentée
            sur plusieurs lignes
        """
        qa_pairs = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Chercher une question (se termine par ::)
            if line.strip().endswith('::'):
                question = line.strip()[:-2]

                # Collecter la réponse indentée
                answer_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]

                    # Si la ligne est indentée, fait partie de la réponse
                    if next_line.startswith((' ', '\t')):
                        answer_lines.append(next_line.strip())
                    elif next_line.strip() == '':
                        # Ligne vide, peut continuer
                        pass
                    else:
                        # Nouvelle question ou contenu non indenté
                        break

                    j += 1

                if answer_lines:
                    qa_pairs.append({
                        'id': f"DL{len(qa_pairs) + 1}",
                        'question': question,
                        'answer': ' '.join(answer_lines),
                        'category': 'Général',
                        'metadata': {'format': 'definition_list'}
                    })

                i = j
            else:
                i += 1

        return qa_pairs

    def _parse_generic_faq(self, content: str) -> List[Dict]:
        """
        Parse générique pour FAQ sans format clair.

        Utilise des heuristiques pour identifier les questions
        (phrases se terminant par ?) et leurs réponses probables.
        """
        qa_pairs = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Chercher une ligne qui ressemble à une question
            is_question = False
            question = ""

            # Vérifier les patterns de question
            for pattern in self.question_starters:
                if re.match(pattern, line):
                    is_question = True
                    # Extraire la question
                    match = re.match(pattern + r'(.+)', line)
                    if match and match.lastindex:
                        question = match.group(match.lastindex).strip()
                    else:
                        question = line
                    break

            # Si pas trouvé avec les starters, chercher les phrases avec ?
            if not is_question and line.endswith('?'):
                is_question = True
                question = line

            if is_question and question:
                # Chercher la réponse
                answer_lines = []
                j = i + 1

                # Stratégie : prendre les lignes suivantes jusqu'à la prochaine question
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Vérifier si c'est une nouvelle question
                    is_next_question = False
                    for pattern in self.question_starters:
                        if re.match(pattern, next_line) or next_line.endswith('?'):
                            is_next_question = True
                            break

                    if is_next_question:
                        break

                    # Vérifier si c'est un marqueur de réponse
                    is_answer_marker = False
                    for pattern in self.answer_starters:
                        if re.match(pattern, next_line):
                            is_answer_marker = True
                            # Retirer le marqueur
                            next_line = re.sub(pattern, '', next_line).strip()
                            break

                    if next_line:
                        answer_lines.append(next_line)
                    elif answer_lines and not is_answer_marker:
                        # Double ligne vide = fin probable de la réponse
                        if j + 1 < len(lines) and not lines[j + 1].strip():
                            break

                    j += 1

                # Ne garder que si on a une réponse
                if answer_lines:
                    qa_pairs.append({
                        'id': f"G{len(qa_pairs) + 1}",
                        'question': question,
                        'answer': ' '.join(answer_lines),
                        'category': 'Général',
                        'metadata': {'format': 'generic', 'confidence': 0.8}
                    })

                    i = j
                else:
                    i += 1
            else:
                i += 1

        return qa_pairs

    def _extract_text_metadata(self, content: str) -> Dict:
        """
        Extrait les métadonnées du texte de la FAQ.
        """
        metadata = {}

        # Chercher les métadonnées dans les premières lignes
        lines = content.split('\n')[:20]  # Regarder les 20 premières lignes

        for line in lines:
            for key, pattern in self.metadata_patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    metadata[key] = match.group(1).strip()

        return metadata

    def _extract_categories_from_text(self, content: str) -> List[Dict]:
        """
        Extrait les catégories/sections du texte.
        """
        categories = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self.category_patterns:
                match = re.match(pattern, line)
                if match:
                    category_name = match.group(1).strip()

                    # Ignorer si c'est probablement une question
                    if not category_name.endswith('?'):
                        categories.append({
                            'name': category_name,
                            'line_number': i,
                            'description': '',
                            'question_count': 0
                        })
                    break

        return categories

    def _assign_categories_to_questions(self, qa_pairs: List[Dict],
                                        categories: List[Dict],
                                        content: str) -> List[Dict]:
        """
        Assigne les catégories aux questions basé sur leur position dans le texte.
        """
        if not categories or not qa_pairs:
            return qa_pairs

        lines = content.split('\n')

        # Pour chaque Q/A, trouver sa position approximative
        for qa in qa_pairs:
            if qa['category'] != 'Général':
                continue  # Déjà catégorisé

            # Trouver la ligne de la question
            question_line = -1
            for i, line in enumerate(lines):
                if qa['question'] in line:
                    question_line = i
                    break

            if question_line == -1:
                continue

            # Trouver la catégorie la plus proche avant cette question
            closest_category = 'Général'
            for cat in reversed(categories):
                if cat['line_number'] < question_line:
                    closest_category = cat['name']
                    cat['question_count'] += 1
                    break

            qa['category'] = closest_category

        return qa_pairs

    def _validate_and_enrich_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Valide et enrichit les paires Q/R extraites.

        Cette méthode :
        - Filtre les paires invalides
        - Nettoie le texte
        - Ajoute des métadonnées utiles
        - Calcule des scores de confiance
        """
        validated_pairs = []

        for qa in qa_pairs:
            # Vérifier la validité de base
            if not qa.get('question') or not qa.get('answer'):
                continue

            # Nettoyer la question et la réponse
            question = self._clean_text(qa['question'])
            answer = self._clean_text(qa['answer'])

            # Vérifier les longueurs minimales
            if len(question.split()) < 2 or len(answer.split()) < 3:
                continue

            # Calculer la confiance
            confidence = self._calculate_qa_confidence(question, answer, qa.get('metadata', {}))

            if confidence >= self.confidence_threshold:
                # Enrichir avec des métadonnées
                enriched_qa = {
                    'id': qa.get('id', f"Q{len(validated_pairs) + 1}"),
                    'question': question,
                    'answer': answer,
                    'category': qa.get('category', 'Général'),
                    'metadata': {
                        **qa.get('metadata', {}),
                        'confidence': confidence,
                        'answer_length': len(answer.split()),
                        'has_keywords': self._contains_security_keywords(question + ' ' + answer)
                    }
                }

                # Détecter les sujets/tags
                enriched_qa['metadata']['topics'] = self._extract_topics(question + ' ' + answer)

                validated_pairs.append(enriched_qa)

        return validated_pairs

    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte en retirant les artifacts indésirables.
        """
        # Retirer les marqueurs Q:/A: restants
        text = re.sub(r'^[QAR]\s*:\s*', '', text)
        text = re.sub(r'^Question\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(?:Answer|Réponse)\s*:\s*', '', text, flags=re.IGNORECASE)

        # Retirer les numéros au début
        text = re.sub(r'^\d+\.\s*', '', text)

        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)

        # Retirer les espaces en début/fin
        text = text.strip()

        return text

    def _calculate_qa_confidence(self, question: str, answer: str, metadata: Dict) -> float:
        """
        Calcule un score de confiance pour une paire Q/R.
        """
        score = 1.0

        # La question doit se terminer par ? (sauf cas spéciaux)
        if not question.endswith('?') and not any(
                starter in question.lower() for starter in ['comment', 'pourquoi', 'what', 'how']):
            score -= 0.2

        # Longueur de la réponse
        answer_length = len(answer.split())
        if answer_length < 5:
            score -= 0.3
        elif answer_length > 500:
            score -= 0.1  # Réponse possiblement trop longue

        # Format détecté avec confiance
        if metadata.get('format') in ['numbered', 'qa_prefixed', 'markdown']:
            score += 0.1

        # Cohérence question/réponse
        # Les mots de la question devraient apparaître dans la réponse
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        common_words = question_words.intersection(answer_words)

        if len(common_words) < 2:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _contains_security_keywords(self, text: str) -> bool:
        """
        Vérifie si le texte contient des mots-clés de cybersécurité.
        """
        security_keywords = [
            'sécurité', 'security', 'authentification', 'authentication',
            'chiffrement', 'encryption', 'mot de passe', 'password',
            'vulnérabilité', 'vulnerability', 'risque', 'risk',
            'conformité', 'compliance', 'audit', 'ISO', 'RGPD', 'GDPR'
        ]

        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in security_keywords)

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extrait les sujets/thèmes principaux du texte.
        """
        topics = []
        text_lower = text.lower()

        # Dictionnaire de topics cybersécurité
        topic_keywords = {
            'authentification': ['authentification', 'authentication', 'login', 'connexion', '2fa', 'mfa'],
            'mots_de_passe': ['mot de passe', 'password', 'mdp', 'credential'],
            'chiffrement': ['chiffrement', 'encryption', 'cryptage', 'aes', 'rsa'],
            'conformité': ['conformité', 'compliance', 'rgpd', 'gdpr', 'iso', 'audit'],
            'incident': ['incident', 'breach', 'violation', 'compromission'],
            'sauvegarde': ['sauvegarde', 'backup', 'restauration', 'recovery'],
            'réseau': ['réseau', 'network', 'firewall', 'pare-feu', 'vpn'],
            'accès': ['accès', 'access', 'permission', 'autorisation', 'privilège']
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics[:5]  # Limiter à 5 topics

    def _organize_by_categories(self, qa_pairs: List[Dict], categories: List[Dict]) -> List[Dict]:
        """
        Organise les Q/R par catégories pour créer des sections.
        """
        sections = []

        # Créer une section par catégorie
        for cat in categories:
            cat_questions = [qa for qa in qa_pairs if qa['category'] == cat['name']]

            if cat_questions:
                section_content = f"## {cat['name']}\n\n"

                for qa in cat_questions:
                    section_content += f"**Q: {qa['question']}**\n\n"
                    section_content += f"R: {qa['answer']}\n\n"

                sections.append({
                    'title': cat['name'],
                    'level': 1,
                    'content': section_content,
                    'question_count': len(cat_questions),
                    'type': 'faq_category'
                })

        # Ajouter les questions sans catégorie
        uncategorized = [qa for qa in qa_pairs if qa['category'] == 'Général']
        if uncategorized:
            section_content = "## Questions Générales\n\n"

            for qa in uncategorized:
                section_content += f"**Q: {qa['question']}**\n\n"
                section_content += f"R: {qa['answer']}\n\n"

            sections.append({
                'title': 'Questions Générales',
                'level': 1,
                'content': section_content,
                'question_count': len(uncategorized),
                'type': 'faq_category'
            })

        return sections

    def _create_flat_sections(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Crée une structure de sections plate quand pas de catégories.
        """
        if not qa_pairs:
            return []

        # Grouper par blocs de 10 questions
        sections = []
        for i in range(0, len(qa_pairs), 10):
            batch = qa_pairs[i:i + 10]

            section_content = ""
            for qa in batch:
                section_content += f"**Q: {qa['question']}**\n\n"
                section_content += f"R: {qa['answer']}\n\n"

            sections.append({
                'title': f'Questions {i + 1} à {min(i + 10, len(qa_pairs))}',
                'level': 1,
                'content': section_content,
                'question_count': len(batch),
                'type': 'faq_batch'
            })

        return sections

    def _generate_structured_content(self, qa_pairs: List[Dict], categories: List[Dict]) -> str:
        """
        Génère le contenu textuel structuré de la FAQ.

        Format optimisé pour le traitement par le système CRAG.
        """
        content_parts = ["# Foire Aux Questions (FAQ)\n"]

        # Résumé
        content_parts.append(f"## Résumé\n")
        content_parts.append(f"- Total des questions : {len(qa_pairs)}")
        content_parts.append(f"- Catégories : {len(categories)}")

        # Topics couverts
        all_topics = set()
        for qa in qa_pairs:
            all_topics.update(qa['metadata'].get('topics', []))

        if all_topics:
            content_parts.append(f"- Sujets couverts : {', '.join(all_topics)}")

        content_parts.append("\n")

        # Organiser par catégorie
        if categories:
            for cat in categories:
                cat_questions = [qa for qa in qa_pairs if qa['category'] == cat['name']]

                if cat_questions:
                    content_parts.append(f"## {cat['name']}\n")

                    for qa in cat_questions:
                        content_parts.append(f"### {qa['question']}\n")
                        content_parts.append(f"{qa['answer']}\n")

                        # Ajouter les métadonnées si pertinentes
                        if qa['metadata'].get('topics'):
                            content_parts.append(f"*Sujets : {', '.join(qa['metadata']['topics'])}*\n")

                        content_parts.append("")
        else:
            # Pas de catégories, liste simple
            content_parts.append("## Questions et Réponses\n")

            for qa in qa_pairs:
                content_parts.append(f"### {qa['question']}\n")
                content_parts.append(f"{qa['answer']}\n")

                if qa['metadata'].get('topics'):
                    content_parts.append(f"*Sujets : {', '.join(qa['metadata']['topics'])}*\n")

                content_parts.append("")

        return "\n".join(content_parts)

    def _create_faq_tables(self, qa_pairs: List[Dict], categories: List[Dict]) -> List[Dict]:
        """
        Crée des tables structurées pour les FAQ.

        Une table par catégorie pour faciliter la recherche.
        """
        tables = []

        if categories:
            for cat in categories:
                cat_questions = [qa for qa in qa_pairs if qa['category'] == cat['name']]

                if cat_questions:
                    table_data = [['ID', 'Question', 'Réponse (résumé)', 'Sujets']]

                    for qa in cat_questions:
                        # Résumer la réponse si trop longue
                        answer_summary = qa['answer'][:100] + '...' if len(qa['answer']) > 100 else qa['answer']
                        topics = ', '.join(qa['metadata'].get('topics', []))

                        table_data.append([
                            qa['id'],
                            qa['question'],
                            answer_summary,
                            topics
                        ])

                    tables.append({
                        'type': 'faq_table',
                        'title': f'FAQ - {cat["name"]}',
                        'data': table_data,
                        'category': cat['name'],
                        'question_count': len(cat_questions)
                    })
        else:
            # Une seule table pour toutes les questions
            table_data = [['ID', 'Question', 'Réponse (résumé)', 'Sujets']]

            for qa in qa_pairs:
                answer_summary = qa['answer'][:100] + '...' if len(qa['answer']) > 100 else qa['answer']
                topics = ', '.join(qa['metadata'].get('topics', []))

                table_data.append([
                    qa['id'],
                    qa['question'],
                    answer_summary,
                    topics
                ])

            tables.append({
                'type': 'faq_table',
                'title': 'FAQ Complète',
                'data': table_data,
                'category': 'Toutes',
                'question_count': len(qa_pairs)
            })

        return tables

    def _find_previous_header(self, element) -> Optional[str]:
        """
        Trouve le header précédent d'un élément HTML (pour BeautifulSoup).
        """
        for sibling in element.find_previous_siblings():
            if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                return sibling.get_text(strip=True)
        return None

    def _find_container_category(self, container) -> Optional[str]:
        """
        Trouve la catégorie d'un container HTML.
        """
        # Chercher dans les classes
        classes = container.get('class', [])
        for cls in classes:
            if 'category' in cls or 'section' in cls:
                # Extraire le nom de la catégorie
                return cls.replace('category-', '').replace('section-', '').replace('-', ' ').title()

        # Chercher un titre dans le container
        header = container.find(['h1', 'h2', 'h3', 'h4'])
        if header:
            return header.get_text(strip=True)

        return None

    def _calculate_confidence_score(self, qa_pairs: List[Dict],
                                    categories: List[Dict],
                                    errors: List[str]) -> float:
        """
        Calcule le score de confiance global du parsing FAQ.
        """
        score = 1.0

        # Pénalités pour les erreurs
        score -= len(errors) * 0.2

        # Vérifier qu'on a trouvé des Q/R
        if not qa_pairs:
            return 0.0
        elif len(qa_pairs) < 3:
            score -= 0.3

        # Bonus pour la structure détectée
        if categories:
            score += 0.1

        # Vérifier la qualité moyenne des Q/R
        avg_confidence = sum(qa['metadata'].get('confidence', 0.5) for qa in qa_pairs) / len(qa_pairs)
        score = score * 0.7 + avg_confidence * 0.3

        # Bonus si beaucoup de questions cybersécurité
        security_questions = sum(1 for qa in qa_pairs if qa['metadata'].get('has_keywords', False))
        if security_questions / len(qa_pairs) > 0.5:
            score += 0.1

        return max(0.0, min(1.0, score))
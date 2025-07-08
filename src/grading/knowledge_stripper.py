import re
from typing import List, Dict, Tuple
import spacy
from dataclasses import dataclass


@dataclass
class KnowledgeStrip:
    """
    Une "bande de connaissance" - un morceau atomique d'information.

    Pense à ça comme une carte de révision : une unité d'information
    qui a du sens toute seule, mais qui fait partie d'un tout plus grand.
    """
    content: str  # Le texte du strip
    strip_type: str  # 'sentence', 'paragraph', 'definition', etc.
    position: int  # Position dans le document original
    context: Dict  # Informations contextuelles
    source_doc_id: int  # ID du document source


class KnowledgeStripper:
    """
    Cette classe découpe intelligemment les documents en unités de connaissance.

    L'idée clé : tous les morceaux d'un document ne sont pas égaux.
    Certains contiennent des définitions, d'autres des exemples,
    d'autres encore des détails techniques.
    """

    def __init__(self, language: str = "fr"):
        # On utilise spaCy pour comprendre la structure linguistique
        try:
            if language == "fr":
                self.nlp = spacy.load("fr_core_news_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(f"⚠️ Modèle spaCy non trouvé. Installe-le avec : python -m spacy download {language}_core_news_sm")
            # Fallback sur une version simple
            self.nlp = None

        # Patterns pour détecter différents types de contenu
        self.definition_patterns = [
            r"(?:est|sont)\s+(?:un|une|des|le|la|les)",  # "X est un/une..."
            r"se définit comme",
            r"consiste à",
            r"représente",
            r"qu'on appelle",
        ]

        self.example_patterns = [
            r"par exemple",
            r"comme",
            r"tel(?:s)?\s+que",
            r"notamment",
            r"illustration",
        ]

    def strip_document(self,
                       document: str,
                       doc_id: int,
                       granularity: str = "adaptive") -> List[KnowledgeStrip]:
        """
        Découpe un document en knowledge strips.

        Args:
            document: Le texte du document
            doc_id: ID du document (pour tracer l'origine)
            granularity:
                - "sentence": Découpe par phrase
                - "paragraph": Découpe par paragraphe
                - "adaptive": Découpe intelligente selon le contenu

        Returns:
            Liste de KnowledgeStrips
        """
        strips = []

        if granularity == "sentence":
            strips = self._strip_by_sentences(document, doc_id)
        elif granularity == "paragraph":
            strips = self._strip_by_paragraphs(document, doc_id)
        else:  # adaptive
            strips = self._adaptive_stripping(document, doc_id)

        return strips

    def _strip_by_sentences(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """Version améliorée qui préserve le contexte."""
        strips = []

        if self.nlp:
            doc = self.nlp(document)
            for i, sent in enumerate(doc.sents):
                # Vérifier si la phrase est complète
                sent_text = sent.text.strip()

                # Si la phrase commence par une minuscule, elle fait probablement
                # partie de la phrase précédente
                if i > 0 and sent_text and sent_text[0].islower():
                    # Fusionner avec la phrase précédente
                    if strips:
                        strips[-1].content += " " + sent_text
                    continue

                strip = KnowledgeStrip(
                    content=sent_text,
                    strip_type="sentence",
                    position=i,
                    context={"char_start": sent.start_char, "char_end": sent.end_char},
                    source_doc_id=doc_id
                )
                strips.append(strip)

        return strips

    def _strip_by_paragraphs(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """Découpe par paragraphes."""
        # Séparer par double retour à la ligne ou indentation
        paragraphs = re.split(r'\n\s*\n', document)

        strips = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                strip = KnowledgeStrip(
                    content=para.strip(),
                    strip_type="paragraph",
                    position=i,
                    context={"paragraph_length": len(para.split())},
                    source_doc_id=doc_id
                )
                strips.append(strip)

        return strips

    def _adaptive_stripping(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Découpage intelligent qui s'adapte au contenu.

        Cette méthode est le cœur de l'innovation :
        - Détecte les définitions et les garde ensemble
        - Identifie les exemples et les lie au concept
        - Préserve les listes et énumérations
        - Groupe les phrases liées sémantiquement
        """
        strips = []

        # D'abord, on fait un découpage par phrases
        sentence_strips = self._strip_by_sentences(document, doc_id)

        # Ensuite, on analyse et regroupe intelligemment
        i = 0
        while i < len(sentence_strips):
            current = sentence_strips[i]

            # Détecter le type de contenu
            content_lower = current.content.lower()

            # Est-ce une définition ?
            if any(re.search(pattern, content_lower) for pattern in self.definition_patterns):
                # Regrouper avec les phrases suivantes qui détaillent
                definition_strip = self._merge_definition_context(sentence_strips, i)
                strips.append(definition_strip)
                i += len(definition_strip.context.get("merged_sentences", [1]))

            # Est-ce un exemple ?
            elif any(re.search(pattern, content_lower) for pattern in self.example_patterns):
                example_strip = self._create_example_strip(sentence_strips, i)
                strips.append(example_strip)
                i += 1

            # Est-ce une énumération ?
            elif re.search(r'^\s*[-•*\d]+[\s.)]', current.content):
                enum_strip = self._merge_enumeration(sentence_strips, i)
                strips.append(enum_strip)
                i += len(enum_strip.context.get("items", [1]))

            else:
                # Phrase normale
                current.strip_type = "statement"
                strips.append(current)
                i += 1

        return strips

    def _merge_definition_context(self,
                                  sentences: List[KnowledgeStrip],
                                  start_idx: int) -> KnowledgeStrip:
        """
        Fusionne une définition avec son contexte immédiat.

        Pourquoi ? Une définition seule peut être incomplète.
        "Les transformers sont des modèles." → Pas très utile
        "Les transformers sont des modèles. Ils utilisent l'attention." → Mieux !
        """
        merged_content = sentences[start_idx].content
        merged_indices = [start_idx]

        # Regarder les 2 phrases suivantes max
        for i in range(1, min(3, len(sentences) - start_idx)):
            next_sent = sentences[start_idx + i].content

            # Si la phrase suivante continue l'explication
            if (next_sent.startswith("Il") or
                    next_sent.startswith("Elle") or
                    next_sent.startswith("Ils") or
                    next_sent.startswith("Elles") or
                    next_sent.startswith("Cela") or
                    next_sent.startswith("Ce")):
                merged_content += " " + next_sent
                merged_indices.append(start_idx + i)
            else:
                break

        return KnowledgeStrip(
            content=merged_content,
            strip_type="definition",
            position=start_idx,
            context={
                "merged_sentences": merged_indices,
                "definition_completeness": len(merged_indices) / 3.0
            },
            source_doc_id=sentences[start_idx].source_doc_id
        )

    def _create_example_strip(self,
                              sentences: List[KnowledgeStrip],
                              idx: int) -> KnowledgeStrip:
        """Crée un strip pour un exemple, en préservant le contexte."""
        return KnowledgeStrip(
            content=sentences[idx].content,
            strip_type="example",
            position=idx,
            context={
                "example_indicator": True,
                "related_concept": sentences[idx - 1].content if idx > 0 else None
            },
            source_doc_id=sentences[idx].source_doc_id
        )

    def _merge_enumeration(self,
                           sentences: List[KnowledgeStrip],
                           start_idx: int) -> KnowledgeStrip:
        """Fusionne les éléments d'une liste ou énumération."""
        items = [sentences[start_idx].content]
        current_idx = start_idx + 1

        # Continuer tant qu'on trouve des items de liste
        while current_idx < len(sentences):
            if re.search(r'^\s*[-•*\d]+[\s.)]', sentences[current_idx].content):
                items.append(sentences[current_idx].content)
                current_idx += 1
            else:
                break

        return KnowledgeStrip(
            content="\n".join(items),
            strip_type="enumeration",
            position=start_idx,
            context={
                "items": list(range(start_idx, current_idx)),
                "item_count": len(items)
            },
            source_doc_id=sentences[start_idx].source_doc_id
        )
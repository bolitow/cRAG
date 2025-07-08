import re
from typing import List, Dict, Tuple
import spacy
from dataclasses import dataclass
from .domain_patterns import CyberSecurityPatterns


@dataclass
class KnowledgeStrip:
    """
    Une "bande de connaissance" - un morceau atomique d'information.

    Pense à ça comme une carte de révision : une unité d'information
    qui a du sens toute seule, mais qui fait partie d'un tout plus grand.
    """
    content: str  # Le texte du strip
    strip_type: str  # Type enrichi pour la cybersécurité
    position: int  # Position dans le document original
    context: Dict  # Informations contextuelles enrichies
    source_doc_id: int  # ID du document source
    domain_category: str  # Nouvelle : catégorie métier (policy, control, etc.)
    compliance_refs: List[str]  # Nouvelle : références normatives détectées


class KnowledgeStripper:
    """
    Cette classe découpe intelligemment les documents en unités de connaissance.

    Version adaptée pour la cybersécurité : elle comprend maintenant
    les structures spécifiques des politiques de sécurité, des procédures,
    et des documents de conformité.
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

        # Initialiser les patterns de cybersécurité
        self.cyber_patterns = CyberSecurityPatterns()

        # Patterns génériques (conservés de la version originale)
        self.definition_patterns = [
            r"(?:est|sont)\s+(?:un|une|des|le|la|les)",
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

        # Nouveaux patterns spécifiques à la cybersécurité
        self.policy_section_markers = [
            r"^\d+\.\d*\s*",  # Numérotation hiérarchique (1.1, 1.2.3, etc.)
            r"^[A-Z]\.\s*",  # Sections alphabétiques
            r"^Article\s+\d+",  # Articles numérotés
            r"^Annexe\s+",  # Annexes
            r"^Appendice\s+"  # Appendices
        ]

    def strip_document(self,
                       document: str,
                       doc_id: int,
                       doc_metadata: Dict = None,
                       granularity: str = "adaptive") -> List[KnowledgeStrip]:
        """
        Découpe un document en knowledge strips avec analyse du domaine.

        Args:
            document: Le texte du document
            doc_id: ID du document (pour tracer l'origine)
            doc_metadata: Métadonnées du document (type, source, date, etc.)
            granularity:
                - "sentence": Découpe par phrase
                - "paragraph": Découpe par paragraphe
                - "section": Découpe par section (nouveau)
                - "adaptive": Découpe intelligente selon le contenu

        Returns:
            Liste de KnowledgeStrips enrichis
        """
        strips = []

        # Déterminer le type de document si pas fourni
        doc_type = self._infer_document_type(document, doc_metadata)

        if granularity == "sentence":
            strips = self._strip_by_sentences(document, doc_id)
        elif granularity == "paragraph":
            strips = self._strip_by_paragraphs(document, doc_id)
        elif granularity == "section":
            strips = self._strip_by_sections(document, doc_id)
        else:  # adaptive
            strips = self._adaptive_stripping_cyber(document, doc_id, doc_type)

        # Post-traitement : enrichir chaque strip avec les métadonnées domaine
        enriched_strips = []
        for strip in strips:
            enriched_strip = self._enrich_strip_with_domain(strip, doc_metadata)
            enriched_strips.append(enriched_strip)

        return enriched_strips

    def _infer_document_type(self, document: str, metadata: Dict = None) -> str:
        """
        Détermine le type de document basé sur son contenu et ses métadonnées.

        Cette fonction est cruciale car elle permet d'adapter le découpage
        selon qu'on traite une politique, une procédure, ou un rapport d'audit.
        """
        # Si le type est fourni dans les métadonnées, l'utiliser
        if metadata and "doc_type" in metadata:
            return metadata["doc_type"]

        # Sinon, analyser le contenu
        doc_lower = document.lower()[:1000]  # Analyser le début du document

        # Indicateurs de type de document
        if any(indicator in doc_lower for indicator in
               ["politique de sécurité", "security policy", "politique générale"]):
            return "policy"
        elif any(indicator in doc_lower for indicator in
                 ["procédure", "procedure", "mode opératoire", "instruction"]):
            return "procedure"
        elif any(indicator in doc_lower for indicator in
                 ["rapport d'audit", "audit report", "compte rendu d'audit"]):
            return "audit_report"
        elif any(indicator in doc_lower for indicator in
                 ["faq", "questions fréquentes", "frequently asked"]):
            return "faq"
        elif any(indicator in doc_lower for indicator in
                 ["matrice", "contrôles", "controls matrix", "registre"]):
            return "control_matrix"
        else:
            return "general"

    def _strip_by_sections(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Nouvelle méthode : découpe par sections logiques.

        Particulièrement utile pour les politiques et procédures qui ont
        une structure hiérarchique claire.
        """
        strips = []

        # Détecter les sections basées sur les marqueurs
        sections = []
        current_section = {"content": "", "header": "", "level": 0}

        lines = document.split('\n')
        for line in lines:
            # Vérifier si c'est un header de section
            is_section_header = False
            for pattern in self.policy_section_markers:
                if re.match(pattern, line.strip()):
                    is_section_header = True
                    # Sauvegarder la section précédente si elle existe
                    if current_section["content"].strip():
                        sections.append(current_section)
                    # Commencer une nouvelle section
                    current_section = {
                        "content": line + "\n",
                        "header": line.strip(),
                        "level": self._determine_section_level(line)
                    }
                    break

            if not is_section_header and line.strip():
                current_section["content"] += line + "\n"

        # Ajouter la dernière section
        if current_section["content"].strip():
            sections.append(current_section)

        # Convertir les sections en strips
        for i, section in enumerate(sections):
            strip = KnowledgeStrip(
                content=section["content"].strip(),
                strip_type="section",
                position=i,
                context={
                    "section_header": section["header"],
                    "section_level": section["level"],
                    "is_structured": True
                },
                source_doc_id=doc_id,
                domain_category="",  # Sera enrichi plus tard
                compliance_refs=[]  # Sera enrichi plus tard
            )
            strips.append(strip)

        return strips

    def _adaptive_stripping_cyber(self,
                                  document: str,
                                  doc_id: int,
                                  doc_type: str) -> List[KnowledgeStrip]:
        """
        Découpage adaptatif spécialisé pour la cybersécurité.

        Cette méthode est le cœur de l'innovation pour votre domaine :
        - Reconnaît les structures de politique (articles, clauses)
        - Identifie les mesures de contrôle et les regroupe
        - Préserve les références normatives
        - Garde ensemble les procédures multi-étapes
        """
        strips = []

        # Stratégie de découpage selon le type de document
        if doc_type == "policy":
            strips = self._strip_policy_document(document, doc_id)
        elif doc_type == "procedure":
            strips = self._strip_procedure_document(document, doc_id)
        elif doc_type == "faq":
            strips = self._strip_faq_document(document, doc_id)
        else:
            # Fallback sur la méthode originale améliorée
            strips = self._enhanced_adaptive_stripping(document, doc_id)

        return strips

    def _strip_policy_document(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Découpage spécialisé pour les documents de politique.

        Les politiques ont généralement :
        - Un préambule/introduction
        - Des articles ou sections numérotées
        - Des obligations (DOIT, NE DOIT PAS)
        - Des références à des normes
        """
        strips = []

        # D'abord, essayer de découper par sections
        section_strips = self._strip_by_sections(document, doc_id)

        # Pour chaque section, analyser si elle doit être subdivisée
        for section_strip in section_strips:
            content = section_strip.content

            # Détecter les déclarations de politique individuelles
            policy_statements = self._extract_policy_statements(content)

            if len(policy_statements) > 1:
                # Subdiviser en statements individuels
                for j, statement in enumerate(policy_statements):
                    strip = KnowledgeStrip(
                        content=statement,
                        strip_type="policy_statement",
                        position=section_strip.position * 100 + j,  # Position hiérarchique
                        context={
                            "parent_section": section_strip.context.get("section_header", ""),
                            "obligation_level": self._determine_obligation_level(statement),
                            "statement_index": j
                        },
                        source_doc_id=doc_id,
                        domain_category="policy",
                        compliance_refs=self.cyber_patterns.extract_compliance_references(statement)
                    )
                    strips.append(strip)
            else:
                # Garder la section entière mais typer correctement
                section_strip.strip_type = "policy_section"
                section_strip.domain_category = "policy"
                strips.append(section_strip)

        return strips

    def _strip_procedure_document(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Découpage spécialisé pour les procédures.

        Les procédures ont généralement :
        - Un objectif/scope
        - Des prérequis
        - Des étapes numérotées
        - Des responsabilités (RACI)
        """
        strips = []

        # Patterns pour identifier les différentes parties d'une procédure
        procedure_parts = {
            "objective": [r"objectif", r"but", r"purpose", r"scope"],
            "prerequisites": [r"prérequis", r"pré-requis", r"prerequisites", r"conditions"],
            "steps": [r"étapes", r"procédure", r"steps", r"process"],
            "responsibilities": [r"responsabilit", r"raci", r"roles"]
        }

        # Découper le document en grandes parties
        lines = document.split('\n')
        current_part = {"type": "general", "content": [], "start_pos": 0}
        parts = []

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Vérifier si on commence une nouvelle partie
            new_part_type = None
            for part_type, patterns in procedure_parts.items():
                if any(re.search(pattern, line_lower) for pattern in patterns):
                    new_part_type = part_type
                    break

            if new_part_type and current_part["content"]:
                # Sauvegarder la partie courante
                parts.append(current_part)
                current_part = {"type": new_part_type, "content": [line], "start_pos": i}
            else:
                current_part["content"].append(line)

        # Ajouter la dernière partie
        if current_part["content"]:
            parts.append(current_part)

        # Convertir en strips
        for part_idx, part in enumerate(parts):
            content = "\n".join(part["content"])

            if part["type"] == "steps":
                # Découper les étapes individuellement
                steps = self._extract_procedure_steps(content)
                for step_idx, step in enumerate(steps):
                    strip = KnowledgeStrip(
                        content=step["content"],
                        strip_type="procedure_step",
                        position=part_idx * 100 + step_idx,
                        context={
                            "step_number": step.get("number", step_idx + 1),
                            "has_substeps": step.get("has_substeps", False),
                            "actors": step.get("actors", [])
                        },
                        source_doc_id=doc_id,
                        domain_category="procedure",
                        compliance_refs=self.cyber_patterns.extract_compliance_references(step["content"])
                    )
                    strips.append(strip)
            else:
                # Garder comme bloc
                strip = KnowledgeStrip(
                    content=content,
                    strip_type=f"procedure_{part['type']}",
                    position=part_idx,
                    context={
                        "part_type": part["type"],
                        "contains_requirements": self._contains_requirements(content)
                    },
                    source_doc_id=doc_id,
                    domain_category="procedure",
                    compliance_refs=self.cyber_patterns.extract_compliance_references(content)
                )
                strips.append(strip)

        return strips

    def _strip_faq_document(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Découpage spécialisé pour les FAQ.

        Structure typique :
        - Question
        - Réponse
        - Parfois des références ou liens
        """
        strips = []

        # Patterns pour détecter les Q&R
        qa_patterns = [
            r"(?:Q|Question)\s*[:.]?\s*(.+?)(?:\n|$).*?(?:R|A|Réponse|Answer)\s*[:.]?\s*(.+?)(?=(?:Q|Question)|$)",
            r"^\s*\d+\.\s*(.+?)\n+(.+?)(?=^\s*\d+\.|$)",  # Numérotées
            r"^#{1,3}\s*(.+?)\n+(.+?)(?=^#{1,3}|$)"  # Markdown headers
        ]

        # Essayer différents patterns
        qa_pairs = []
        for pattern in qa_patterns:
            matches = re.finditer(pattern, document, re.MULTILINE | re.DOTALL)
            qa_pairs = list(matches)
            if qa_pairs:
                break

        if qa_pairs:
            for i, match in enumerate(qa_pairs):
                question = match.group(1).strip()
                answer = match.group(2).strip()

                # Créer un strip pour chaque Q&R
                strip = KnowledgeStrip(
                    content=f"Question: {question}\n\nRéponse: {answer}",
                    strip_type="faq_entry",
                    position=i,
                    context={
                        "question": question,
                        "answer": answer,
                        "topics": self._extract_topics(question + " " + answer)
                    },
                    source_doc_id=doc_id,
                    domain_category="faq",
                    compliance_refs=self.cyber_patterns.extract_compliance_references(answer)
                )
                strips.append(strip)
        else:
            # Fallback : découper par paragraphes
            strips = self._strip_by_paragraphs(document, doc_id)
            for strip in strips:
                strip.strip_type = "faq_content"
                strip.domain_category = "faq"

        return strips

    def _enhanced_adaptive_stripping(self, document: str, doc_id: int) -> List[KnowledgeStrip]:
        """
        Version améliorée du stripping adaptatif qui intègre la connaissance domaine.
        """
        strips = []

        # D'abord, on fait un découpage par phrases
        sentence_strips = self._strip_by_sentences(document, doc_id)

        i = 0
        while i < len(sentence_strips):
            current = sentence_strips[i]
            content_lower = current.content.lower()

            # Analyser le type de contenu avec les patterns cyber
            domain_category = self.cyber_patterns.get_pattern_category(current.content)

            # Appliquer une stratégie de regroupement selon la catégorie
            if domain_category == "policy":
                # Regrouper les déclarations de politique connexes
                merged_strip = self._merge_policy_context(sentence_strips, i)
                merged_strip.strip_type = "policy_statement"
                merged_strip.domain_category = "policy"
                strips.append(merged_strip)
                i += len(merged_strip.context.get("merged_sentences", [1]))

            elif domain_category == "control":
                # Identifier le type de contrôle
                control_types = self.cyber_patterns.identify_control_types(current.content)
                merged_strip = self._merge_control_description(sentence_strips, i)
                merged_strip.strip_type = "control_measure"
                merged_strip.domain_category = "control"
                merged_strip.context["control_types"] = control_types
                strips.append(merged_strip)
                i += len(merged_strip.context.get("merged_sentences", [1]))

            elif domain_category == "compliance":
                # Extraire les références normatives
                compliance_refs = self.cyber_patterns.extract_compliance_references(current.content)
                merged_strip = self._merge_compliance_context(sentence_strips, i)
                merged_strip.strip_type = "compliance_requirement"
                merged_strip.domain_category = "compliance"
                merged_strip.compliance_refs = compliance_refs
                strips.append(merged_strip)
                i += len(merged_strip.context.get("merged_sentences", [1]))

            elif domain_category == "procedure":
                # Garder les étapes de procédure ensemble
                merged_strip = self._merge_procedure_steps(sentence_strips, i)
                merged_strip.strip_type = "procedure_step"
                merged_strip.domain_category = "procedure"
                strips.append(merged_strip)
                i += len(merged_strip.context.get("merged_sentences", [1]))

            elif domain_category == "risk":
                # Regrouper l'analyse de risque
                current.strip_type = "risk_statement"
                current.domain_category = "risk"
                strips.append(current)
                i += 1

            else:
                # Utiliser la logique originale pour les cas généraux
                # (définitions, exemples, etc.)
                if any(re.search(pattern, content_lower) for pattern in self.definition_patterns):
                    definition_strip = self._merge_definition_context(sentence_strips, i)
                    strips.append(definition_strip)
                    i += len(definition_strip.context.get("merged_sentences", [1]))
                elif any(re.search(pattern, content_lower) for pattern in self.example_patterns):
                    example_strip = self._create_example_strip(sentence_strips, i)
                    strips.append(example_strip)
                    i += 1
                elif re.search(r'^\s*[-•*\d]+[\s.)]', current.content):
                    enum_strip = self._merge_enumeration(sentence_strips, i)
                    strips.append(enum_strip)
                    i += len(enum_strip.context.get("items", [1]))
                else:
                    current.strip_type = "statement"
                    strips.append(current)
                    i += 1

        return strips

    def _enrich_strip_with_domain(self,
                                  strip: KnowledgeStrip,
                                  doc_metadata: Dict = None) -> KnowledgeStrip:
        """
        Enrichit un strip avec des métadonnées spécifiques au domaine.

        Cette fonction ajoute des informations contextuelles importantes
        pour l'évaluation de la pertinence et la génération de réponses.
        """
        # Si pas déjà fait, déterminer la catégorie domaine
        if not strip.domain_category:
            strip.domain_category = self.cyber_patterns.get_pattern_category(strip.content)

        # Si pas déjà fait, extraire les références de conformité
        if not strip.compliance_refs:
            strip.compliance_refs = self.cyber_patterns.extract_compliance_references(strip.content)

        # Ajouter des métadonnées du document si disponibles
        if doc_metadata:
            strip.context["doc_type"] = doc_metadata.get("doc_type", "unknown")
            strip.context["doc_version"] = doc_metadata.get("version", "")
            strip.context["doc_date"] = doc_metadata.get("date", "")
            strip.context["confidentiality"] = doc_metadata.get("confidentiality", "internal")
            strip.context["validity_period"] = doc_metadata.get("validity_period", "")

        # Ajouter un score d'autorité basé sur le type de document
        authority_scores = {
            "policy": 1.0,  # Les politiques ont l'autorité maximale
            "standard": 0.9,  # Les standards sont très autoritaires
            "procedure": 0.8,  # Les procédures sont autoritaires
            "guideline": 0.6,  # Les guides sont moins stricts
            "faq": 0.5,  # Les FAQ sont informationnelles
            "general": 0.3  # Les documents généraux ont peu d'autorité
        }

        doc_type = doc_metadata.get("doc_type", "general") if doc_metadata else "general"
        strip.context["authority_score"] = authority_scores.get(doc_type, 0.3)

        # Marquer les strips critiques pour la sécurité
        critical_keywords = [
            "critique", "critical", "urgent", "immédiat", "immediate",
            "obligatoire", "mandatory", "interdit", "forbidden", "prohibited"
        ]

        if any(keyword in strip.content.lower() for keyword in critical_keywords):
            strip.context["is_critical"] = True
            strip.context["criticality_level"] = "high"

        return strip

    def _extract_policy_statements(self, content: str) -> List[str]:
        """
        Extrait les déclarations de politique individuelles d'un texte.
        """
        statements = []

        # Patterns pour identifier les déclarations
        statement_patterns = [
            r"(?:^|\n)(?:\d+\.?\s*)?(?:Il est |L'organisation |La société |Nous |Le personnel ).*?(?:doit|doivent|ne doit pas|ne doivent pas).*?(?:\.|;|\n\n)",
            r"(?:^|\n)(?:\d+\.?\s*)?Tout.*?(?:doit|doivent|est|sont).*?(?:\.|;|\n\n)",
            r"(?:^|\n)(?:\d+\.?\s*)?\w+.*?(?:shall|must|shall not|must not).*?(?:\.|;|\n\n)"
        ]

        for pattern in statement_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                statement = match.group(0).strip()
                if len(statement.split()) >= 5:  # Au moins 5 mots
                    statements.append(statement)

        # Si pas de statements trouvés, retourner le contenu entier
        if not statements:
            statements = [content]

        return statements

    def _determine_obligation_level(self, statement: str) -> str:
        """
        Détermine le niveau d'obligation d'une déclaration de politique.
        """
        statement_lower = statement.lower()

        if any(word in statement_lower for word in
               ["doit", "doivent", "obligatoire", "impératif", "must", "shall", "mandatory"]):
            return "mandatory"
        elif any(word in statement_lower for word in
                 ["ne doit pas", "ne doivent pas", "interdit", "prohibition", "must not", "shall not"]):
            return "forbidden"
        elif any(word in statement_lower for word in
                 ["devrait", "devraient", "recommandé", "should", "recommended"]):
            return "recommended"
        elif any(word in statement_lower for word in
                 ["peut", "peuvent", "optionnel", "may", "can", "optional"]):
            return "optional"
        else:
            return "informational"

    def _extract_procedure_steps(self, content: str) -> List[Dict]:
        """
        Extrait les étapes individuelles d'une procédure.
        """
        steps = []

        # Patterns pour détecter les étapes
        step_patterns = [
            r"(?:Étape|Step)\s+(\d+)\s*[:\-.]?\s*(.+?)(?=(?:Étape|Step)\s+\d+|$)",
            r"(\d+)\.\s+(.+?)(?=\d+\.|$)",
            r"([a-z])\)\s+(.+?)(?=[a-z]\)|$)"
        ]

        for pattern in step_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            if matches:
                for match in matches:
                    step_content = match.group(2).strip() if match.lastindex >= 2 else match.group(0)
                    step = {
                        "number": match.group(1),
                        "content": step_content,
                        "has_substeps": bool(re.search(r'[a-z]\)', step_content)),
                        "actors": self._extract_actors(step_content)
                    }
                    steps.append(step)
                break

        # Si pas d'étapes trouvées, considérer le tout comme une étape
        if not steps:
            steps = [{"number": 1, "content": content, "has_substeps": False, "actors": []}]

        return steps

    def _extract_actors(self, content: str) -> List[str]:
        """
        Extrait les acteurs/rôles mentionnés dans un texte.
        """
        actors = []

        # Patterns pour identifier les acteurs
        actor_patterns = [
            r"(?:Le |La |L')(\w+(?:\s+\w+)?)\s+(?:doit|peut|vérifie|approuve|valide)",
            r"(?:RSSI|DSI|DPO|administrateur|utilisateur|responsable|manager|auditeur)",
            r"(?:security officer|admin|user|manager|auditor)"
        ]

        for pattern in actor_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                actor = match.group(1) if match.lastindex else match.group(0)
                if actor and actor.lower() not in ["le", "la", "l'"]:
                    actors.append(actor)

        return list(set(actors))  # Éliminer les doublons

    def _extract_topics(self, content: str) -> List[str]:
        """
        Extrait les sujets/thèmes principaux d'un texte.
        """
        topics = []

        # Utiliser les mots-clés cyber pour identifier les topics
        content_lower = content.lower()

        # Vérifier les frameworks de conformité
        for framework in self.cyber_patterns.compliance_frameworks:
            if framework.lower() in content_lower:
                topics.append(f"compliance:{framework}")

        # Vérifier les types de contrôles
        for control_type, keywords in self.cyber_patterns.control_keywords.items():
            for keyword in keywords[:3]:  # Prendre seulement les premiers
                if keyword in content_lower:
                    topics.append(f"control:{control_type}")
                    break

        # Vérifier les concepts de risque
        for risk_keyword in self.cyber_patterns.risk_keywords.get("risk_type", []):
            if risk_keyword in content_lower:
                topics.append(f"risk:{risk_keyword}")
                break

        return list(set(topics))[:5]  # Limiter à 5 topics

    def _merge_policy_context(self, sentences: List[KnowledgeStrip], start_idx: int) -> KnowledgeStrip:
        """
        Fusionne une déclaration de politique avec son contexte.
        """
        merged_content = sentences[start_idx].content
        merged_indices = [start_idx]

        # Regarder les phrases suivantes pour le contexte
        for i in range(1, min(3, len(sentences) - start_idx)):
            next_sent = sentences[start_idx + i].content

            # Si la phrase suivante développe la politique
            if (any(word in next_sent.lower() for word in
                    ["cela", "cette mesure", "cette politique", "ce qui"]) or
                    any(word in next_sent.lower() for word in
                        ["en particulier", "notamment", "incluant", "y compris"])):
                merged_content += " " + next_sent
                merged_indices.append(start_idx + i)
            else:
                break

        return KnowledgeStrip(
            content=merged_content,
            strip_type="policy_statement",
            position=start_idx,
            context={
                "merged_sentences": merged_indices,
                "completeness": len(merged_indices) / 3.0,
                "obligation_level": self._determine_obligation_level(merged_content)
            },
            source_doc_id=sentences[start_idx].source_doc_id,
            domain_category="policy",
            compliance_refs=[]
        )

    def _merge_control_description(self, sentences: List[KnowledgeStrip], start_idx: int) -> KnowledgeStrip:
        """
        Fusionne une description de contrôle avec ses détails.
        """
        merged_content = sentences[start_idx].content
        merged_indices = [start_idx]

        # Chercher les détails d'implémentation
        for i in range(1, min(4, len(sentences) - start_idx)):
            next_sent = sentences[start_idx + i].content.lower()

            # Indicateurs de détails de contrôle
            if any(indicator in next_sent for indicator in
                   ["configuration", "paramètre", "setting", "mise en œuvre",
                    "implementation", "déploiement", "deployment"]):
                merged_content += " " + sentences[start_idx + i].content
                merged_indices.append(start_idx + i)
            elif re.search(r'^\s*[-•]', sentences[start_idx + i].content):
                # Points de configuration
                merged_content += "\n" + sentences[start_idx + i].content
                merged_indices.append(start_idx + i)
            else:
                break

        return KnowledgeStrip(
            content=merged_content,
            strip_type="control_measure",
            position=start_idx,
            context={
                "merged_sentences": merged_indices,
                "has_implementation_details": len(merged_indices) > 1
            },
            source_doc_id=sentences[start_idx].source_doc_id,
            domain_category="control",
            compliance_refs=[]
        )

    def _merge_compliance_context(self, sentences: List[KnowledgeStrip], start_idx: int) -> KnowledgeStrip:
        """
        Fusionne une exigence de conformité avec ses détails.
        """
        merged_content = sentences[start_idx].content
        merged_indices = [start_idx]

        # Chercher les détails de mise en conformité
        for i in range(1, min(3, len(sentences) - start_idx)):
            next_sent = sentences[start_idx + i].content

            # Si la phrase suivante détaille l'exigence
            if any(word in next_sent.lower() for word in
                   ["pour satisfaire", "afin de", "cela implique", "nécessite",
                    "to comply", "this requires", "implementation"]):
                merged_content += " " + next_sent
                merged_indices.append(start_idx + i)
            else:
                break

        return KnowledgeStrip(
            content=merged_content,
            strip_type="compliance_requirement",
            position=start_idx,
            context={
                "merged_sentences": merged_indices,
                "requirement_completeness": len(merged_indices) / 2.0
            },
            source_doc_id=sentences[start_idx].source_doc_id,
            domain_category="compliance",
            compliance_refs=[]
        )

    def _merge_procedure_steps(self, sentences: List[KnowledgeStrip], start_idx: int) -> KnowledgeStrip:
        """
        Fusionne les étapes de procédure connexes.
        """
        merged_content = sentences[start_idx].content
        merged_indices = [start_idx]

        # Chercher les sous-étapes ou précisions
        for i in range(1, min(5, len(sentences) - start_idx)):
            next_sent = sentences[start_idx + i].content

            # Patterns de continuation d'étape
            if (re.search(r'^\s*[a-z]\)', next_sent) or  # Sous-étapes a) b) c)
                    re.search(r'^\s*[-•]', next_sent) or  # Points
                    any(word in next_sent.lower() for word in
                        ["ensuite", "puis", "après", "then", "next"])):
                merged_content += "\n" + next_sent
                merged_indices.append(start_idx + i)
            else:
                break

        return KnowledgeStrip(
            content=merged_content,
            strip_type="procedure_step",
            position=start_idx,
            context={
                "merged_sentences": merged_indices,
                "is_multi_step": len(merged_indices) > 1,
                "step_count": len(merged_indices)
            },
            source_doc_id=sentences[start_idx].source_doc_id,
            domain_category="procedure",
            compliance_refs=[]
        )

    def _determine_section_level(self, line: str) -> int:
        """
        Détermine le niveau hiérarchique d'une section.
        """
        line = line.strip()

        # Compter les points dans la numérotation (1.2.3 = niveau 3)
        if re.match(r'^\d+(\.\d+)*', line):
            return line.count('.') + 1
        # Headers markdown
        elif line.startswith('#'):
            return len(line) - len(line.lstrip('#'))
        # Articles/Sections
        elif line.startswith('Article'):
            return 1
        elif line.startswith('Section'):
            return 2
        else:
            return 0

    def _contains_requirements(self, content: str) -> bool:
        """
        Vérifie si le contenu contient des exigences/obligations.
        """
        requirement_keywords = [
            "doit", "doivent", "obligatoire", "requis", "nécessaire",
            "must", "shall", "required", "mandatory"
        ]

        content_lower = content.lower()
        return any(keyword in content_lower for keyword in requirement_keywords)

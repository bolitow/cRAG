import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
from .audit_relevance_rules import AuditRelevanceAnalyzer, AuditContext


class RelevanceScore(Enum):
    """
    Niveaux de pertinence pour un knowledge strip.
    On utilise une échelle simple mais efficace.
    """
    HIGHLY_RELEVANT = 0.9  # Répond directement à la question
    RELEVANT = 0.7  # Contient des infos utiles
    SOMEWHAT_RELEVANT = 0.5  # Contexte utile mais indirect
    BARELY_RELEVANT = 0.3  # Mention tangentielle
    NOT_RELEVANT = 0.1  # Aucun rapport


@dataclass
class GradedStrip:
    """Un strip avec son score de pertinence et l'explication."""
    strip: 'KnowledgeStrip'
    relevance_score: float
    confidence: float
    reasoning: str
    relevance_category: RelevanceScore
    audit_context: AuditContext


class RelevanceGrader:
    """
    Le grader évalue la pertinence de chaque knowledge strip par rapport à une requête.

    Il combine plusieurs approches :
    1. Similarité sémantique (via embeddings)
    2. Analyse des mots-clés et concepts
    3. Modèle de classification (si disponible)
    4. Heuristiques linguistiques
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 use_llm: bool = True,
                 embedder=None,
                 enable_audit_rules: bool = True):  # Nouveau paramètre
        """
        Args:
            model_name: Nom du modèle pour le grading
            use_llm: Utiliser un LLM pour le grading
            embedder: Embedder pour calculer les similarités
            enable_audit_rules: Activer les règles spécifiques aux audits
        """

        self.use_llm = use_llm
        self.embedder = embedder

        if use_llm and model_name:
            print(f"Chargement du modèle de grading : {model_name}")
            try:
                # Pour Gemma ou autre petit modèle
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = pipeline(
                    "text-generation",
                    model=model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle : {e}")
                print("Fallback sur les heuristiques")
                self.use_llm = False

        # Initialiser l'analyseur d'audit si activé
        self.enable_audit_rules = enable_audit_rules
        if self.enable_audit_rules:
            self.audit_analyzer = AuditRelevanceAnalyzer()

        # Patterns pour détecter différents types de pertinence
        self._init_patterns()
        # Initialiser les relations conceptuelles
        self._init_concept_relations()

    def _init_patterns(self):
        """Initialise les patterns pour l'analyse heuristique."""
        # Mots indiquant une définition ou explication
        self.definition_indicators = [
            "est", "sont", "signifie", "définit", "représente",
            "consiste", "permet", "utilise", "fonctionne"
        ]

        # Mots indiquant un exemple
        self.example_indicators = [
            "par exemple", "comme", "tel que", "notamment",
            "illustration", "instance", "cas"
        ]

        # Mots de connexion importants
        self.causal_indicators = [
            "parce que", "car", "donc", "ainsi", "en effet",
            "grâce à", "permet de", "conduit à"
        ]

        # Mots indiquant un mécanisme ou processus
        self.mechanism_indicators = [
            "mécanisme", "processus", "méthode", "technique",
            "utilise", "emploie", "applique", "traite", "analyse"
        ]

    def _init_concept_relations(self):
        """
        Initialise les relations entre concepts.
        C'est crucial pour comprendre que parler d'attention,
        c'est parler du fonctionnement des transformers.
        """
        self.concept_relations = {
            "transformers": {
                "mécanismes_clés": ["attention", "self-attention", "auto-attention",
                                    "multi-head", "multi-têtes", "positional encoding",
                                    "encodage positionnel"],
                "composants": ["encoder", "decoder", "encodeur", "décodeur",
                               "feed-forward", "couches"],
                "caractéristiques": ["parallélisation", "parallèle", "contexte",
                                     "bidirectionnel", "séquence"],
                "modèles_associés": ["bert", "gpt", "t5", "bart"]
            },
            "fonctionnent": {
                "synonymes": ["marche", "opère", "fonctionne", "travaille"],
                "aspects": ["mécanisme", "processus", "architecture", "structure",
                            "utilise", "emploie", "applique"]
            },
            "réseaux de neurones": {
                "types": ["cnn", "rnn", "lstm", "transformer", "convolutif", "récurrent"],
                "concepts": ["couche", "neurone", "activation", "poids", "biais"]
            }
        }

    def _extract_main_concept(self, query: str) -> Optional[str]:
        """Extrait le concept principal de la requête."""
        query_lower = query.lower()
        for concept in self.concept_relations.keys():
            if concept in query_lower:
                return concept
        return None

    def grade_strips(self,
                     query: str,
                     strips: List['KnowledgeStrip'],
                     detailed: bool = True,
                     audit_metadata: Optional[Dict] = None) -> List[GradedStrip]:
        """
        Évalue la pertinence de chaque strip par rapport à la requête.

        Version améliorée avec prise en compte du contexte d'audit.

        Args:
            query: La question/requête de l'utilisateur
            strips: Liste des knowledge strips à évaluer
            detailed: Si True, fournit un raisonnement détaillé
            audit_metadata: Métadonnées sur le contexte d'audit (nouveau)

        Returns:
            Liste de GradedStrips triés par pertinence
        """
        graded_strips = []

        # Détecter le contexte d'audit si les règles sont activées
        audit_context = None
        if self.enable_audit_rules and self.audit_analyzer:
            audit_context = self.audit_analyzer.detect_audit_context(query, audit_metadata)
            print(f"Contexte d'audit détecté : {audit_context.value}")

        # Extraire les concepts clés de la requête
        query_concepts = self._extract_key_concepts(query)
        main_concept = self._extract_main_concept(query)

        # Si contexte d'audit, enrichir avec les mots-clés spécifiques
        if audit_context and self.enable_audit_rules:
            audit_keywords = self.audit_analyzer.get_audit_specific_keywords(audit_context)
            # Ajouter les mots-clés critiques aux concepts
            for keyword in audit_keywords.get('critical', []):
                query_concepts[keyword] = 1.2  # Poids élevé pour les éléments critiques

        for strip in strips:
            # Calculer plusieurs scores de pertinence
            scores = {}

            # 1. Similarité sémantique (si embedder disponible)
            if self.embedder:
                scores['semantic'] = self._compute_semantic_similarity(query, strip.content)

            # 2. Overlap de mots-clés amélioré
            scores['keyword'] = self._compute_keyword_overlap_enhanced(
                query_concepts, strip.content, query, main_concept
            )

            # 3. Type de contenu amélioré
            scores['content_type'] = self._score_by_content_type_enhanced(query, strip)

            # 4. Pertinence conceptuelle (nouveau)
            scores['conceptual'] = self._compute_conceptual_relevance(
                query, strip.content, main_concept
            )

            # 5. Score LLM (si disponible)
            if self.use_llm:
                scores['llm'] = self._compute_llm_relevance(query, strip.content)

            # NOUVEAU : Score d'audit si activé
            if self.enable_audit_rules and audit_context:
                # Le score de base est la moyenne des autres scores
                base_score = sum(scores.values()) / len(scores) if scores else 0.5

                # Calculer le bonus d'audit
                audit_adjusted_score, audit_explanation = self.audit_analyzer.calculate_audit_relevance_bonus(
                    query=query,
                    strip=strip,
                    base_score=base_score,
                    metadata=audit_metadata
                )

                scores['audit'] = audit_adjusted_score

                # Stocker l'explication pour le raisonnement
                if 'audit_reasoning' not in strip.context:
                    strip.context['audit_reasoning'] = audit_explanation

            # Combiner les scores avec des poids ajustés
            final_score, confidence = self._combine_scores_enhanced(scores, audit_context)

            # Post-processing pour éviter les anomalies
            final_score = self._post_process_score(final_score, strip, query, main_concept)

            # Déterminer la catégorie
            category = self._categorize_relevance(final_score)

            # Générer le raisonnement
            reasoning = self._generate_reasoning(query, strip, scores, detailed)

            graded_strips.append(GradedStrip(
                strip=strip,
                relevance_score=final_score,
                confidence=confidence,
                reasoning=reasoning,
                relevance_category=category,
                audit_context=audit_context  # Nouveau champ
            ))

        # Trier par score décroissant
        graded_strips.sort(key=lambda x: x.relevance_score, reverse=True)

        # Si contexte d'audit, vérifier la complétude
        if self.enable_audit_rules and audit_context:
            self._check_audit_completeness(query, graded_strips, audit_context, audit_metadata)

        return graded_strips

    def _extract_key_concepts(self, query: str) -> Dict[str, float]:
        """
        Extrait les concepts clés de la requête avec leur importance.

        Par exemple, pour "Comment fonctionnent les transformers ?":
        - "fonctionnent" → 0.8 (verbe principal)
        - "transformers" → 1.0 (sujet principal)
        - "comment" → 0.3 (mot interrogatif)
        """
        # Tokenisation simple
        words = query.lower().split()

        # Filtrer les mots vides
        stop_words = {"le", "la", "les", "un", "une", "des", "de", "du", "?", "!", "."}
        words = [w for w in words if w not in stop_words]

        # Donner plus de poids aux noms et verbes (heuristique simple)
        concepts = {}
        for word in words:
            if word in ["comment", "pourquoi", "quoi", "où", "quand"]:
                concepts[word] = 0.3
            elif word.endswith("ent"):  # Verbes conjugués probable
                concepts[word] = 0.8
                # Ajouter aussi la forme infinitive probable
                if word == "fonctionnent":
                    concepts["fonctionne"] = 0.8
                    concepts["fonctionner"] = 0.8
            else:  # Noms et autres
                concepts[word] = 1.0

        return concepts

    def _compute_semantic_similarity(self, query: str, content: str) -> float:
        """Calcule la similarité sémantique via embeddings."""
        if not self.embedder:
            return 0.5

        query_emb = self.embedder.embed_text(query)[0]
        content_emb = self.embedder.embed_text(content)[0]

        similarity = self.embedder.compute_similarity(query_emb, content_emb)

        # Normaliser entre 0 et 1
        return (similarity + 1) / 2

    def _compute_keyword_overlap_enhanced(self,
                                          query_concepts: Dict[str, float],
                                          content: str,
                                          query: str,
                                          main_concept: Optional[str]) -> float:
        """
        Version améliorée du calcul de chevauchement des mots-clés.

        Cette version comprend les relations conceptuelles :
        - Si on cherche "transformers", "attention" est pertinent
        - Si on cherche "fonctionnent", "mécanisme" est pertinent
        """
        content_lower = content.lower()
        query_lower = query.lower()  # FIX: Définir query_lower
        total_weight = sum(query_concepts.values())
        matched_weight = 0

        # Dictionnaire pour tracker ce qui a déjà été trouvé (éviter les doublons)
        found_concepts = set()

        # D'abord, vérifier les concepts directs
        for concept, weight in query_concepts.items():
            concept_lower = concept.lower()

            # Recherche exacte
            if concept_lower in content_lower and concept_lower not in found_concepts:
                matched_weight += weight
                found_concepts.add(concept_lower)
            # Recherche de variantes (simple stemming)
            elif concept_lower not in found_concepts:
                if concept_lower.rstrip('s') in content_lower or concept_lower + 's' in content_lower:
                    matched_weight += weight * 0.8
                    found_concepts.add(concept_lower)
                # Recherche de la racine
                elif len(concept_lower) > 4 and concept_lower[:4] in content_lower:
                    matched_weight += weight * 0.5
                    found_concepts.add(concept_lower)

        # Entités importantes avec leur poids
        important_entities = {
            # Modèles
            "bert": 2.0,
            "gpt": 2.0,
            "t5": 2.0,
            "bart": 2.0,
            # Architectures
            "transformer": 1.5,
            "transformers": 1.5,
            "attention": 1.5,
            "encoder": 1.2,
            "decoder": 1.2,
            # Concepts clés
            "bidirectional": 1.0,
            "autoregressive": 1.0,
            "multi-head": 1.0,
            "multi-têtes": 1.0
        }

        # Vérifier les entités importantes
        for entity, entity_weight in important_entities.items():
            # Si l'entité est dans la question ET dans le contenu
            if entity in query_lower and entity in content_lower:
                # Bonus proportionnel à l'importance de l'entité
                matched_weight += entity_weight

                # Si c'est l'entité principale de la question, bonus supplémentaire
                if main_concept and entity in main_concept.lower():
                    matched_weight += entity_weight * 0.5

        # Ensuite, vérifier les concepts liés
        if main_concept and main_concept in self.concept_relations:
            related_concepts = self.concept_relations[main_concept]
            for category, terms in related_concepts.items():
                category_matched = False
                for term in terms:
                    if term.lower() in content_lower and not category_matched:
                        # Les termes liés valent 70% du poids du concept principal
                        matched_weight += query_concepts.get(main_concept, 1.0) * 0.7
                        category_matched = True  # Un seul match par catégorie
                        break

        # Cas spéciaux pour des questions spécifiques
        # Si on demande "Qu'est-ce que BERT" et que le contenu parle de BERT
        if "bert" in query_lower and "bert" in content_lower:
            # Vérifier si c'est une vraie explication de BERT
            bert_indicators = [
                "bidirectional encoder",
                "encoder representations",
                "pré-entraîné",
                "pre-trained",
                "masqués",
                "masked"
            ]
            if any(indicator in content_lower for indicator in bert_indicators):
                matched_weight += 3.0  # Gros bonus pour une vraie description de BERT

        # Même chose pour GPT
        if "gpt" in query_lower and "gpt" in content_lower:
            gpt_indicators = [
                "generative",
                "pre-trained",
                "autoregressive",
                "decoder",
                "génère"
            ]
            if any(indicator in content_lower for indicator in gpt_indicators):
                matched_weight += 3.0

        # Calculer le score de base
        base_score = matched_weight / total_weight if total_weight > 0 else 0

        # Ajustements finaux
        # Si le score est très élevé (> 2.0), c'est probablement très pertinent
        if base_score > 2.0:
            return 1.0  # Score maximum
        elif base_score > 1.0:
            # Ramener sur une échelle 0-1 avec une courbe qui favorise les hauts scores
            return 0.8 + (0.2 * (base_score - 1.0))  # Entre 0.8 et 1.0
        else:
            return base_score

    def _score_by_content_type_enhanced(self, query: str, strip: 'KnowledgeStrip') -> float:
        """
        Version améliorée qui comprend mieux les différents types de questions.

        Notamment, "comment fonctionne" devrait favoriser :
        - Les définitions qui expliquent le mécanisme
        - Les descriptions de processus
        - Les énumérations de composants
        """
        query_lower = query.lower()

        # Pour "comment fonctionne(nt)"
        if "comment" in query_lower and any(word in query_lower for word in ["fonctionne", "marche"]):
            # Les mécanismes sont très pertinents
            if any(ind in strip.content.lower() for ind in self.mechanism_indicators):
                return 0.9
            # Les définitions qui expliquent sont pertinentes
            if strip.strip_type == "definition":
                return 0.8
            # Les énumérations de composants sont utiles
            if strip.strip_type == "enumeration" and any(
                    word in strip.content.lower() for word in ["composant", "élément", "partie"]
            ):
                return 0.7
            # Les explications causales sont très pertinentes
            if any(ind in strip.content.lower() for ind in self.causal_indicators):
                return 0.85

        # Pour "qu'est-ce que"
        elif "qu'est-ce" in query_lower or "quoi" in query_lower or "définition" in query_lower:
            if strip.strip_type == "definition":
                return 0.9
            elif any(ind in strip.content.lower() for ind in self.definition_indicators):
                return 0.7

        # Pour "pourquoi"
        elif "pourquoi" in query_lower:
            if any(ind in strip.content.lower() for ind in self.causal_indicators):
                return 0.9

        # Pour "exemple"
        elif "exemple" in query_lower:
            if strip.strip_type == "example":
                return 0.9
            elif any(ind in strip.content.lower() for ind in self.example_indicators):
                return 0.8

        # Score par défaut selon le type
        type_scores = {
            "definition": 0.6,
            "statement": 0.5,
            "example": 0.4,
            "enumeration": 0.5
        }

        return type_scores.get(strip.strip_type, 0.5)

    def _compute_conceptual_relevance(self,
                                      query: str,
                                      content: str,
                                      main_concept: Optional[str]) -> float:
        """
        Nouvelle méthode qui évalue la pertinence conceptuelle.

        Elle comprend que certains termes sont intrinsèquement liés :
        - "attention" est un mécanisme clé des "transformers"
        - "convolution" est lié aux "CNN"
        """
        if not main_concept:
            return 0.5

        content_lower = content.lower()
        relevance_score = 0.0

        # Vérifier si le contenu parle du concept principal
        if main_concept.lower() in content_lower:
            relevance_score = 0.7

        # Vérifier les concepts fortement liés
        if main_concept in self.concept_relations:
            # Mécanismes clés = très pertinent
            if "mécanismes_clés" in self.concept_relations[main_concept]:
                for mechanism in self.concept_relations[main_concept]["mécanismes_clés"]:
                    if mechanism.lower() in content_lower:
                        relevance_score = max(relevance_score, 0.9)
                        break

            # Composants = pertinent
            if "composants" in self.concept_relations[main_concept]:
                for component in self.concept_relations[main_concept]["composants"]:
                    if component.lower() in content_lower:
                        relevance_score = max(relevance_score, 0.7)
                        break

        # Pénaliser si on parle d'un concept concurrent
        # (ex: CNN quand on cherche des transformers)
        for concept, relations in self.concept_relations.items():
            if concept != main_concept and concept in content_lower:
                if "types" in relations:  # C'est une catégorie
                    # Vérifier si notre concept principal n'est pas dans cette catégorie
                    if main_concept not in relations.get("types", []):
                        relevance_score *= 0.3  # Forte pénalité

        return relevance_score

    def _compute_llm_relevance(self, query: str, content: str) -> float:
        """
        Utilise le LLM pour évaluer la pertinence.

        C'est ici qu'on peut utiliser Gemma ou un autre modèle.
        """
        if not self.use_llm:
            return 0.5

        # Prompt simple mais efficace
        prompt = f"""Évalue la pertinence de ce texte pour répondre à la question.
Question : {query}
Texte : {content}

Réponds uniquement par un score entre 0 et 10, où :
- 0-3 : Non pertinent
- 4-6 : Moyennement pertinent  
- 7-8 : Pertinent
- 9-10 : Très pertinent

Score :"""

        try:
            response = self.model(prompt, max_new_tokens=10)
            # Extraire le score de la réponse
            score_text = response[0]['generated_text'].split("Score :")[-1].strip()
            score = float(re.search(r'\d+', score_text).group()) / 10
            return score
        except:
            # En cas d'erreur, retourner un score neutre
            return 0.5

    def _combine_scores_enhanced(self, scores: Dict[str, float], audit_context=None) -> Tuple[float, float]:
        """
        Version améliorée de la combinaison des scores avec prise en compte du contexte d'audit.
        """
        if not scores:
            return 0.5, 0.0

        # Pondérations adaptatives selon le contexte
        if audit_context and 'audit' in scores:
            # En contexte d'audit, le score d'audit est prépondérant
            weights = {
                'audit': 0.40,  # Le plus important
                'keyword': 0.20,  # Toujours important
                'conceptual': 0.15,  # Concepts liés
                'content_type': 0.15,  # Type de contenu
                'semantic': 0.10  # Moins important en audit
            }
        else:
            # Pondérations originales si pas de contexte d'audit
            weights = {
                'semantic': 0.20,
                'keyword': 0.30,
                'content_type': 0.20,
                'conceptual': 0.25,
                'llm': 0.05
            }

        # Calculer la moyenne pondérée
        total_weight = 0
        weighted_sum = 0

        for score_type, score in scores.items():
            if score_type in weights:
                weighted_sum += score * weights[score_type]
                total_weight += weights[score_type]

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Calculer la confiance (plus élevée si on a un contexte d'audit clair)
        if len(scores) > 1:
            variance = np.var(list(scores.values()))
            confidence = 1 - (variance * 2)

            # Bonus de confiance si contexte d'audit détecté
            if audit_context and 'audit' in scores:
                confidence += 0.1

            confidence = max(0.1, min(1.0, confidence))
        else:
            confidence = 0.5

        return final_score, confidence

    def _check_audit_completeness(self,
                                  query: str,
                                  graded_strips: List[GradedStrip],
                                  audit_context: AuditContext,
                                  audit_metadata: Optional[Dict]):
        """
        Vérifie si les strips trouvés sont suffisants pour répondre à la question d'audit.

        Cette méthode ajoute des métadonnées sur la complétude de la réponse.
        """
        if not self.audit_analyzer:
            return

        # Obtenir les suggestions d'éléments manquants
        suggestions = self.audit_analyzer.suggest_missing_elements(
            query=query,
            found_strips=graded_strips,
            context=audit_context
        )

        # Stocker les suggestions dans les métadonnées du premier strip
        # (ou créer un strip spécial pour les métadonnées)
        if graded_strips and suggestions:
            graded_strips[0].strip.context['audit_completeness'] = {
                'is_complete': len(suggestions) == 0,
                'missing_elements': suggestions,
                'audit_context': audit_context.value
            }

    def _post_process_score(self,
                            score: float,
                            strip: 'KnowledgeStrip',
                            query: str,
                            main_concept: Optional[str]) -> float:
        """
        Post-traitement pour corriger les anomalies évidentes.

        Par exemple :
        - Un strip sur les CNN ne devrait jamais avoir un score élevé
          pour une question sur les transformers
        - Un strip qui mentionne juste une date n'est pas très utile
          pour comprendre un fonctionnement
        """
        content_lower = strip.content.lower()

        # Détecter les cas de concepts complètement différents
        if main_concept == "transformers":
            if any(term in content_lower for term in ["cnn", "convolutif", "convolution"]):
                # Ce n'est clairement pas pertinent
                score = min(score, 0.3)

        # Détecter les contenus trop courts ou peu informatifs
        if len(strip.content.split()) < 5:
            score *= 0.7  # Pénalité pour contenu trop court

        # Bonus pour les contenus qui expliquent vraiment
        if "comment" in query.lower() and "fonctionne" in query.lower():
            if any(word in content_lower for word in ["utilise", "permet", "mécanisme", "processus"]):
                score = min(score * 1.2, 1.0)  # Bonus mais plafonné à 1.0

        # Pénalité pour les contenus purement historiques quand on demande le fonctionnement
        if "comment" in query.lower() or "fonctionnement" in query.lower():
            if re.search(r'\b(19|20)\d{2}\b', content_lower) and len(strip.content.split()) < 20:
                score *= 0.5  # Probablement juste une date, pas une explication

        return score

    def _categorize_relevance(self, score: float) -> RelevanceScore:
        """Catégorise le score en niveau de pertinence."""
        if score >= 0.8:
            return RelevanceScore.HIGHLY_RELEVANT
        elif score >= 0.65:
            return RelevanceScore.RELEVANT
        elif score >= 0.5:
            return RelevanceScore.SOMEWHAT_RELEVANT
        elif score >= 0.3:
            return RelevanceScore.BARELY_RELEVANT
        else:
            return RelevanceScore.NOT_RELEVANT

    def _generate_reasoning(self,
                            query: str,
                            strip: 'KnowledgeStrip',
                            scores: Dict[str, float],
                            detailed: bool) -> str:
        """
        Version améliorée qui inclut le raisonnement d'audit.
        """
        if not detailed:
            return f"Score global : {sum(scores.values()) / len(scores):.2f}"

        reasoning_parts = []

        # Analyser chaque composante du score
        if 'semantic' in scores:
            reasoning_parts.append(f"Similarité sémantique : {scores['semantic']:.2f}")

        if 'keyword' in scores:
            reasoning_parts.append(f"Mots-clés : {scores['keyword']:.2f}")

        if 'conceptual' in scores and scores['conceptual'] > 0.5:
            reasoning_parts.append(f"Pertinence conceptuelle : {scores['conceptual']:.2f}")

        # Ajouter le raisonnement d'audit si disponible
        if 'audit' in scores and 'audit_reasoning' in strip.context:
            reasoning_parts.append(f"Audit : {strip.context['audit_reasoning']}")

        if 'content_type' in scores:
            type_msg = f"Type '{strip.strip_type}' "
            if scores['content_type'] > 0.7:
                type_msg += "adapté"
            elif scores['content_type'] > 0.5:
                type_msg += "moyen"
            else:
                type_msg += "peu adapté"
            reasoning_parts.append(type_msg)

        return " | ".join(reasoning_parts)
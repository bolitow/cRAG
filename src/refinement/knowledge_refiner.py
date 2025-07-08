from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import re


@dataclass
class RefinedKnowledge:
    """
    Représente le résultat du processus de raffinement.
    C'est le contexte optimisé qui sera envoyé au générateur.
    """
    strips: List['GradedStrip']  # Les strips sélectionnés et ordonnés
    total_strips: int  # Nombre initial de strips
    retained_strips: int  # Nombre de strips gardés
    coverage_score: float  # Score de couverture de la question
    coverage_details: Dict[str, any]  # Détails de ce qui est couvert
    missing_aspects: List[str]  # Aspects de la question non couverts
    needs_additional_search: bool  # Faut-il chercher plus d'info ?
    summary: str  # Résumé du processus de raffinement


class KnowledgeRefiner:
    """
    Knowledge Refiner équilibré : intelligent et pragmatique.

    Cette version comprend le sens du contenu sans être trop exigeante.
    Elle reconnaît qu'une bonne réponse peut prendre plusieurs formes.
    """

    def __init__(self,
                 relevance_threshold: float = 0.5,
                 similarity_threshold: float = 0.85,
                 max_strips: int = 10,
                 min_coverage: float = 0.6,  # Réduit de 0.7
                 embedder=None):
        """
        Args:
            relevance_threshold: Score minimum pour garder un strip
            similarity_threshold: Seuil de similarité pour la déduplication
            max_strips: Nombre maximum de strips à garder
            min_coverage: Couverture minimale requise (ajustée pour être réaliste)
            embedder: Embedder pour l'analyse sémantique
        """
        self.relevance_threshold = relevance_threshold
        self.similarity_threshold = similarity_threshold
        self.max_strips = max_strips
        self.min_coverage = min_coverage
        self.embedder = embedder

        # Initialiser les composants d'analyse
        self._init_semantic_patterns()
        self._init_question_intents()

    def _init_semantic_patterns(self):
        """
        Initialise les patterns pour l'analyse sémantique.
        Version pragmatique qui reconnaît diverses façons d'exprimer les concepts.
        """
        # Patterns pour détecter différents types d'information
        self.information_patterns = {
            "mécanisme": {
                "keywords": ["mécanisme", "fonctionne", "utilise", "processus",
                             "méthode", "technique", "principe", "clé"],
                "patterns": [
                    r"est le.*clé",
                    r"utilise.*pour",
                    r"permet.*de",
                    r"calcule.*entre",
                    r"traite.*les",
                    r"fonctionne.*par",
                    r"basé.*sur"
                ]
            },
            "processus": {
                "keywords": ["processus", "étape", "phase", "séquence", "calcule",
                             "traite", "analyse", "transforme"],
                "patterns": [
                    r"d'abord.*ensuite",
                    r"commence.*par",
                    r"entre.*tous.*les",
                    r"pour.*chaque",
                    r"séquence.*de"
                ]
            },
            "architecture": {
                "keywords": ["architecture", "structure", "composant", "élément",
                             "module", "couche", "partie", "organisation"],
                "patterns": [
                    r"composé.*de",
                    r"contient.*:",
                    r"inclut.*:",
                    r"divisé.*en",
                    r"-\s+.*\n-",  # Listes
                    r"•\s+.*\n•"  # Listes avec bullets
                ]
            },
            "exemple": {
                "keywords": ["exemple", "comme", "tel que", "notamment",
                             "bert", "gpt", "illustration"],
                "patterns": [
                    r"par exemple",
                    r"comme\s+\w+",
                    r"tel.*que",
                    r"notamment",
                    r"(bert|gpt|t5)\s+"
                ]
            }
        }

        # Relations d'inférence simplifiées
        self.inference_rules = {
            "si_mécanisme_alors_processus": {
                "condition": ["calcule", "traite", "analyse", "transforme"],
                "implique": "description_processus"
            },
            "si_liste_alors_architecture": {
                "condition": [r"-\s+", r"•\s+", r"\d+\."],
                "implique": "architecture_système"
            },
            "si_modèle_spécifique_alors_exemple": {
                "condition": ["bert", "gpt", "t5", "transformer"],
                "implique": "exemple_concret"
            }
        }

    def _init_question_intents(self):
        """
        Définition pragmatique des intentions de questions.
        Moins de besoins, plus de flexibilité.
        """
        self.question_intents = {
            "comment_fonctionne": {
                "patterns": [
                    r"comment.*fonctionne",
                    r"comment.*marche",
                    r"comment.*(?:fait|font)",
                    r"comment.*utilise"
                ],
                "besoins_essentiels": [
                    "explication_mécanisme",
                    "description_architecture"
                ],
                "besoins_optionnels": [
                    "exemple_concret"
                ]
            },
            "qu_est_ce_que": {
                "patterns": [
                    r"qu'est[- ]ce",
                    r"c'est quoi",
                    r"définition",
                    r"que sont"
                ],
                "besoins_essentiels": [
                    "définition_claire"
                ],
                "besoins_optionnels": [
                    "caractéristiques",
                    "exemple_concret"
                ]
            },
            "pourquoi": {
                "patterns": [
                    r"pourquoi",
                    r"quelle.*raison",
                    r"quel.*but"
                ],
                "besoins_essentiels": [
                    "justification"
                ],
                "besoins_optionnels": [
                    "avantages"
                ]
            }
        }

    def refine_knowledge(self,
                         graded_strips: List['GradedStrip'],
                         query: str,
                         detailed_analysis: bool = True) -> RefinedKnowledge:
        """
        Processus de raffinement équilibré et pragmatique.
        """
        print(f"\n=== KNOWLEDGE REFINEMENT (Version Équilibrée) ===")
        print(f"Strips initiaux : {len(graded_strips)}")

        # Étape 1 : Filtrage intelligent mais pas trop strict
        filtered_strips = self._filter_by_relevance(graded_strips)
        print(f"Après filtrage : {len(filtered_strips)} strips")

        # Étape 2 : Déduplication
        deduplicated_strips = self._deduplicate_strips(filtered_strips)
        print(f"Après déduplication : {len(deduplicated_strips)} strips")

        # Étape 3 : Analyse de la question
        question_analysis = self._analyze_question(query)
        print(f"Type de question : {question_analysis['intent']}")

        # Étape 4 : Évaluation pragmatique de la couverture
        coverage_analysis = self._evaluate_coverage_pragmatic(
            deduplicated_strips,
            question_analysis
        )
        print(f"Couverture : {coverage_analysis['score']:.1%}")

        # Étape 5 : Organisation finale
        final_strips = self._organize_strips_by_narrative(
            deduplicated_strips,
            question_analysis
        )

        # Étape 6 : Décision finale
        needs_more = self._decide_if_needs_more(
            coverage_analysis,
            final_strips,
            question_analysis
        )

        # Créer le résumé
        summary = self._create_summary(
            len(graded_strips),
            len(final_strips),
            coverage_analysis,
            needs_more
        )

        return RefinedKnowledge(
            strips=final_strips,
            total_strips=len(graded_strips),
            retained_strips=len(final_strips),
            coverage_score=coverage_analysis['score'],
            coverage_details=coverage_analysis,
            missing_aspects=coverage_analysis.get('missing', []),
            needs_additional_search=needs_more,
            summary=summary
        )

    def _filter_by_relevance(self, graded_strips: List['GradedStrip']) -> List['GradedStrip']:
        """
        Filtrage pragmatique qui garde la diversité d'information.
        """
        filtered = []
        info_types_count = defaultdict(int)

        for strip in graded_strips:
            # Garder tous les strips au-dessus du seuil
            if strip.relevance_score >= self.relevance_threshold:
                filtered.append(strip)
                info_types_count[strip.strip.strip_type] += 1

            # Garder certains strips en dessous du seuil s'ils apportent de la diversité
            elif strip.relevance_score >= 0.4:
                # Garder les définitions (toujours utiles)
                if strip.strip.strip_type == "definition":
                    filtered.append(strip)
                # Garder au moins un exemple
                elif strip.strip.strip_type == "example" and info_types_count["example"] == 0:
                    filtered.append(strip)
                # Garder les énumérations (souvent des composants)
                elif strip.strip.strip_type == "enumeration":
                    filtered.append(strip)

        return filtered

    def _deduplicate_strips(self, strips: List['GradedStrip']) -> List['GradedStrip']:
        """
        Déduplication simple mais efficace.
        """
        if not strips:
            return []

        deduplicated = []
        seen_signatures = set()

        for strip in sorted(strips, key=lambda x: x.relevance_score, reverse=True):
            # Créer une signature simple du contenu
            signature = self._create_content_signature(strip.strip.content)

            # Vérifier si c'est vraiment un doublon
            if signature not in seen_signatures:
                # Vérifier aussi la similarité avec ce qu'on a déjà
                is_duplicate = False

                if self.embedder and len(deduplicated) > 0:
                    strip_emb = self.embedder.embed_text(strip.strip.content)[0]

                    for existing in deduplicated:
                        existing_emb = self.embedder.embed_text(existing.strip.content)[0]
                        similarity = self.embedder.compute_similarity(strip_emb, existing_emb)

                        # Seuil élevé pour éviter de perdre des infos utiles
                        if similarity > self.similarity_threshold:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    deduplicated.append(strip)
                    seen_signatures.add(signature)

        return deduplicated

    def _create_content_signature(self, content: str) -> str:
        """
        Crée une signature simple pour détecter les doublons évidents.
        """
        # Extraire les mots significatifs
        words = re.findall(r'\b\w{4,}\b', content.lower())
        # Garder les 5 mots les plus longs (souvent les plus significatifs)
        significant_words = sorted(words, key=len, reverse=True)[:5]
        return " ".join(sorted(significant_words))

    def _analyze_question(self, query: str) -> Dict[str, any]:
        """
        Analyse pragmatique de la question.
        """
        query_lower = query.lower()
        analysis = {
            "query": query,
            "intent": "general",
            "besoins": [],
            "concepts_clés": []
        }

        # Détecter l'intention
        for intent, config in self.question_intents.items():
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    analysis["intent"] = intent
                    # Combiner besoins essentiels et optionnels
                    analysis["besoins"] = (
                            config["besoins_essentiels"] +
                            config.get("besoins_optionnels", [])
                    )
                    break

        # Extraire les concepts clés (mots importants)
        words = query_lower.split()
        stop_words = {"le", "la", "les", "un", "une", "des", "de", "?", "comment", "qu'est-ce"}
        concepts = [w for w in words if len(w) > 3 and w not in stop_words]
        analysis["concepts_clés"] = concepts

        return analysis

    def _evaluate_coverage_pragmatic(self,
                                     strips: List['GradedStrip'],
                                     question_analysis: Dict) -> Dict[str, any]:
        """
        Évaluation pragmatique qui cherche l'essence, pas la forme.
        """
        coverage = {
            "score": 0.0,
            "aspects_couverts": [],
            "missing": [],
            "détails": {}
        }

        # Concaténer tout le contenu
        all_content = "\n".join([s.strip.content.lower() for s in strips])

        # Vérifier chaque besoin de manière flexible
        for besoin in question_analysis["besoins"]:
            is_covered = self._check_if_need_is_covered(besoin, all_content, strips)

            if is_covered:
                coverage["aspects_couverts"].append(besoin)
            else:
                # Seulement marquer comme manquant si c'est essentiel
                if besoin in question_analysis.get("besoins_essentiels", []):
                    coverage["missing"].append(besoin)

        # Calculer le score de manière pragmatique
        total_besoins = len(question_analysis.get("besoins_essentiels", ["général"]))
        if total_besoins > 0:
            # On compte seulement les besoins essentiels pour le score
            essentiels_couverts = sum(
                1 for b in coverage["aspects_couverts"]
                if b in question_analysis.get("besoins_essentiels", [])
            )
            coverage["score"] = essentiels_couverts / total_besoins
        else:
            # Pas de besoins spécifiques = on regarde la qualité générale
            if len(strips) >= 3 and any(s.relevance_score > 0.7 for s in strips):
                coverage["score"] = 0.8
            else:
                coverage["score"] = 0.5

        # Bonus si on a beaucoup de contenu de qualité
        high_quality_count = sum(1 for s in strips if s.relevance_score >= 0.7)
        if high_quality_count >= 3:
            coverage["score"] = min(1.0, coverage["score"] * 1.2)

        # Vérifier aussi les concepts clés
        concepts_found = sum(
            1 for concept in question_analysis["concepts_clés"]
            if concept in all_content
        )
        if question_analysis["concepts_clés"]:
            concept_coverage = concepts_found / len(question_analysis["concepts_clés"])
            # Moyenne pondérée avec la couverture des besoins
            coverage["score"] = (coverage["score"] * 0.7) + (concept_coverage * 0.3)

        return coverage

    def _check_if_need_is_covered(self,
                                  besoin: str,
                                  content: str,
                                  strips: List['GradedStrip']) -> bool:
        """
        Vérifie de manière flexible si un besoin est couvert.
        """
        # Mapping des besoins vers les patterns de détection
        detection_rules = {
            "explication_mécanisme": {
                "any_of": [
                    "mécanisme",
                    "utilise",
                    "permet",
                    "fonctionne",
                    "calcule",
                    "traite",
                    "processus"
                ],
                "patterns": [
                    r"est le.*clé",
                    r"utilise.*pour",
                    r"permet.*de"
                ]
            },
            "description_architecture": {
                "any_of": [
                    "architecture",
                    "structure",
                    "composant",
                    "couche",
                    "module",
                    "élément"
                ],
                "patterns": [
                    r"-\s+\w+",  # Listes
                    r"composé.*de",
                    r"inclut"
                ],
                "strip_types": ["enumeration"]
            },
            "définition_claire": {
                "any_of": [
                    "est un",
                    "sont des",
                    "représente",
                    "consiste"
                ],
                "strip_types": ["definition"]
            },
            "exemple_concret": {
                "any_of": [
                    "exemple",
                    "comme",
                    "bert",
                    "gpt",
                    "tel que"
                ],
                "patterns": [
                    r"(bert|gpt|t5)",
                    r"par exemple"
                ]
            }
        }

        # Vérifier si le besoin est dans nos règles
        if besoin not in detection_rules:
            # Si pas de règle spécifique, vérifier si le mot apparaît
            return besoin.replace("_", " ") in content

        rules = detection_rules[besoin]

        # Vérifier les mots-clés
        if "any_of" in rules:
            for keyword in rules["any_of"]:
                if keyword in content:
                    return True

        # Vérifier les patterns
        if "patterns" in rules:
            for pattern in rules["patterns"]:
                if re.search(pattern, content):
                    return True

        # Vérifier les types de strips
        if "strip_types" in rules:
            for strip in strips:
                if strip.strip.strip_type in rules["strip_types"]:
                    return True

        return False

    def _organize_strips_by_narrative(self,
                                      strips: List['GradedStrip'],
                                      question_analysis: Dict) -> List['GradedStrip']:
        """
        Organisation simple mais efficace pour créer une narration cohérente.
        """
        if not strips:
            return []

        organized = []
        used = set()

        # Phase 1 : Définitions et introductions
        for i, strip in enumerate(strips):
            if i not in used and strip.strip.strip_type == "definition":
                organized.append(strip)
                used.add(i)

        # Phase 2 : Mécanismes et explications principales
        for i, strip in enumerate(strips):
            if i not in used:
                content_lower = strip.strip.content.lower()
                if any(word in content_lower for word in
                       ["mécanisme", "utilise", "permet", "calcule", "traite"]):
                    organized.append(strip)
                    used.add(i)

        # Phase 3 : Détails et composants
        for i, strip in enumerate(strips):
            if i not in used and strip.strip.strip_type == "enumeration":
                organized.append(strip)
                used.add(i)

        # Phase 4 : Exemples
        for i, strip in enumerate(strips):
            if i not in used:
                content_lower = strip.strip.content.lower()
                if any(word in content_lower for word in ["exemple", "bert", "gpt"]):
                    organized.append(strip)
                    used.add(i)

        # Phase 5 : Le reste par pertinence
        remaining = [(i, strip) for i, strip in enumerate(strips) if i not in used]
        remaining.sort(key=lambda x: x[1].relevance_score, reverse=True)
        organized.extend([strip for _, strip in remaining])

        # Limiter au maximum
        return organized[:self.max_strips]

    def _decide_if_needs_more(self,
                              coverage: Dict,
                              strips: List['GradedStrip'],
                              question_analysis: Dict) -> bool:
        """
        Décision pragmatique : a-t-on assez d'info pour répondre ?
        """
        # Si excellente couverture, pas besoin de plus
        if coverage["score"] >= 0.8:
            return False

        # Si on a beaucoup de contenu de qualité, c'est probablement suffisant
        high_quality_count = sum(1 for s in strips if s.relevance_score >= 0.7)
        if high_quality_count >= 4:
            return False

        # Si on a au moins couvrir les besoins essentiels
        essentiels = question_analysis.get("besoins_essentiels", [])
        if essentiels:
            essentiels_manquants = [
                b for b in essentiels
                if b not in coverage["aspects_couverts"]
            ]
            # OK si on a couvert au moins la moitié des essentiels
            if len(essentiels_manquants) <= len(essentiels) / 2:
                return False

        # Si on a peu de contenu
        if len(strips) < 3:
            return True

        # Si la couverture est vraiment faible
        if coverage["score"] < self.min_coverage:
            return True

        # Par défaut, on fait confiance à ce qu'on a
        return False

    def _create_summary(self,
                        initial_count: int,
                        final_count: int,
                        coverage: Dict,
                        needs_more: bool) -> str:
        """
        Résumé clair et informatif du raffinement.
        """
        parts = []

        # Stats de base
        parts.append(f"Raffinement : {initial_count} → {final_count} strips")
        parts.append(f"Couverture : {coverage['score']:.0%}")

        # Ce qui est couvert
        if coverage["aspects_couverts"]:
            # Traduire en termes lisibles
            aspects_lisibles = []
            for aspect in coverage["aspects_couverts"][:3]:
                if "mécanisme" in aspect:
                    aspects_lisibles.append("mécanisme")
                elif "architecture" in aspect:
                    aspects_lisibles.append("structure")
                elif "définition" in aspect:
                    aspects_lisibles.append("définition")
                elif "exemple" in aspect:
                    aspects_lisibles.append("exemples")

            if aspects_lisibles:
                parts.append(f"Contient : {', '.join(set(aspects_lisibles))}")

        # Recommandation
        if needs_more:
            parts.append("⚠️ Info supplémentaire utile")
        else:
            parts.append("✓ Info suffisante")

        return " | ".join(parts)
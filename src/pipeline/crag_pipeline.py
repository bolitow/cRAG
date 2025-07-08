from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
from datetime import datetime

# Imports de tous nos composants
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import BaseRetriever
from src.grading.knowledge_stripper import KnowledgeStripper
from src.grading.relevance_grader import RelevanceGrader
from src.refinement.knowledge_refiner import KnowledgeRefiner
from src.generation.response_generator import ResponseGenerator, GeneratedResponse


@dataclass
class CRAGResult:
    """
    Résultat complet du pipeline CRAG.
    Contient la réponse et toutes les métadonnées du processus.
    """
    query: str
    answer: str
    confidence: float
    processing_time: float
    steps_details: Dict[str, any]
    needs_more_info: bool
    debug_info: Optional[Dict] = None


class CRAGPipeline:
    """
    Pipeline Corrective RAG complet.

    Ce pipeline orchestre tout le processus :
    1. Retrieval : Trouve les documents pertinents
    2. Stripping : Découpe en unités de connaissance
    3. Grading : Évalue la pertinence
    4. Refinement : Filtre et organise
    5. Generation : Produit la réponse finale

    C'est le chef d'orchestre qui fait travailler tous les composants ensemble.
    """

    def __init__(self,
                 embedder: Optional[DocumentEmbedder] = None,
                 retriever: Optional[BaseRetriever] = None,
                 stripper: Optional[KnowledgeStripper] = None,
                 grader: Optional[RelevanceGrader] = None,
                 refiner: Optional[KnowledgeRefiner] = None,
                 generator: Optional[ResponseGenerator] = None,
                 verbose: bool = True):
        """
        Initialise le pipeline avec tous ses composants.
        Si un composant n'est pas fourni, il sera créé avec les paramètres par défaut.
        """
        self.verbose = verbose

        # Initialiser les composants
        self._log("🚀 Initialisation du pipeline CRAG...")

        # Embedder (partagé entre plusieurs composants)
        self.embedder = embedder or DocumentEmbedder()

        # Retriever
        self.retriever = retriever or BaseRetriever(self.embedder)

        # Knowledge Stripper
        self.stripper = stripper or KnowledgeStripper()

        # Relevance Grader
        self.grader = grader or RelevanceGrader(
            use_llm=False,  # Par défaut sans LLM
            embedder=self.embedder
        )

        # Knowledge Refiner
        self.refiner = refiner or KnowledgeRefiner(
            embedder=self.embedder,
            relevance_threshold=0.5,
            min_coverage=0.6
        )

        # Response Generator
        self.generator = generator or ResponseGenerator(
            use_templates=True,
            response_style="educational"
        )

        self._log("✅ Pipeline CRAG initialisé avec succès!")

    def index_documents(self,
                        documents: List[str],
                        metadata: Optional[List[Dict]] = None,
                        batch_size: int = 32):
        """
        Indexe des documents dans la base de connaissances.

        C'est la préparation : on stocke les documents pour pouvoir
        les retrouver plus tard quand on aura des questions.
        """
        self._log(f"📚 Indexation de {len(documents)} documents...")
        start_time = time.time()

        self.retriever.index_documents(documents, metadata, batch_size)

        indexing_time = time.time() - start_time
        self._log(f"✅ Indexation terminée en {indexing_time:.2f}s")

    def process_query(self,
                      query: str,
                      max_strips: int = 10,
                      debug: bool = False) -> CRAGResult:
        """
        Traite une requête de bout en bout.

        C'est ici que la magie opère : la question entre,
        une réponse structurée et vérifiée sort.
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"🔍 NOUVELLE REQUÊTE : {query}")
        self._log(f"{'=' * 60}")

        start_time = time.time()
        steps_details = {}

        try:
            # Étape 1 : Retrieval
            retrieved_docs = self._step_retrieval(query, steps_details)

            # Étape 2 : Knowledge Stripping
            all_strips = self._step_stripping(retrieved_docs, steps_details)

            # Étape 3 : Relevance Grading
            graded_strips = self._step_grading(query, all_strips, steps_details)

            # Étape 4 : Knowledge Refinement
            refined_knowledge = self._step_refinement(query, graded_strips, steps_details)

            # Étape 5 : Response Generation
            response = self._step_generation(query, refined_knowledge, steps_details)

            # Calculer le temps total
            processing_time = time.time() - start_time

            # Créer le résultat final
            result = CRAGResult(
                query=query,
                answer=response.answer,
                confidence=response.confidence,
                processing_time=processing_time,
                steps_details=steps_details,
                needs_more_info=refined_knowledge.needs_additional_search,
                debug_info=self._create_debug_info(
                    retrieved_docs,
                    graded_strips,
                    refined_knowledge,
                    response
                ) if debug else None
            )

            self._log_final_summary(result)

            return result

        except Exception as e:
            self._log(f"❌ ERREUR : {str(e)}", error=True)
            # Retourner un résultat d'erreur
            return CRAGResult(
                query=query,
                answer=f"Désolé, une erreur s'est produite : {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time,
                steps_details=steps_details,
                needs_more_info=True
            )

    def _step_retrieval(self, query: str, steps_details: Dict) -> List[Dict]:
        """
        Étape 1 : Récupération des documents pertinents.
        """
        self._log("\n📚 ÉTAPE 1 : RETRIEVAL")
        start = time.time()

        # Récupérer les documents
        retrieved_docs = self.retriever.retrieve(query, k=5)

        # Enregistrer les détails
        steps_details["retrieval"] = {
            "duration": time.time() - start,
            "docs_retrieved": len(retrieved_docs),
            "top_scores": [doc["score"] for doc in retrieved_docs[:3]]
        }

        self._log(f"   → {len(retrieved_docs)} documents récupérés")
        for i, doc in enumerate(retrieved_docs[:3]):
            self._log(f"   → Doc {i + 1} (score: {doc['score']:.3f}): {doc['text'][:50]}...")

        return retrieved_docs

    def _step_stripping(self, retrieved_docs: List[Dict], steps_details: Dict) -> List:
        """
        Étape 2 : Découpage en knowledge strips.
        """
        self._log("\n✂️  ÉTAPE 2 : KNOWLEDGE STRIPPING")
        start = time.time()

        all_strips = []
        strip_counts = []

        for doc in retrieved_docs:
            strips = self.stripper.strip_document(
                doc['text'],
                doc['index'],
                granularity="adaptive"
            )
            all_strips.extend(strips)
            strip_counts.append(len(strips))

        steps_details["stripping"] = {
            "duration": time.time() - start,
            "total_strips": len(all_strips),
            "strips_per_doc": strip_counts
        }

        self._log(f"   → {len(all_strips)} strips extraits au total")
        self._log(f"   → Distribution : {strip_counts}")

        return all_strips

    def _step_grading(self, query: str, all_strips: List, steps_details: Dict) -> List:
        """
        Étape 3 : Évaluation de la pertinence.
        """
        self._log("\n⭐ ÉTAPE 3 : RELEVANCE GRADING")
        start = time.time()

        # Grader tous les strips
        graded_strips = self.grader.grade_strips(query, all_strips)

        # Analyser la distribution des scores
        score_distribution = {
            "HIGHLY_RELEVANT": sum(1 for s in graded_strips if s.relevance_score >= 0.8),
            "RELEVANT": sum(1 for s in graded_strips if 0.65 <= s.relevance_score < 0.8),
            "SOMEWHAT_RELEVANT": sum(1 for s in graded_strips if 0.5 <= s.relevance_score < 0.65),
            "LOW_RELEVANCE": sum(1 for s in graded_strips if s.relevance_score < 0.5)
        }

        steps_details["grading"] = {
            "duration": time.time() - start,
            "total_graded": len(graded_strips),
            "score_distribution": score_distribution,
            "top_scores": [s.relevance_score for s in graded_strips[:5]]
        }

        self._log(f"   → Distribution des scores : {score_distribution}")
        self._log(f"   → Top 3 strips :")
        for i, strip in enumerate(graded_strips[:3]):
            self._log(f"      {i + 1}. [{strip.relevance_category.name}] {strip.strip.content[:60]}...")

        return graded_strips

    def _step_refinement(self, query: str, graded_strips: List, steps_details: Dict):
        """
        Étape 4 : Raffinement et organisation.
        """
        self._log("\n🔧 ÉTAPE 4 : KNOWLEDGE REFINEMENT")
        start = time.time()

        # Raffiner les connaissances
        refined_knowledge = self.refiner.refine_knowledge(graded_strips, query)

        steps_details["refinement"] = {
            "duration": time.time() - start,
            "strips_before": len(graded_strips),
            "strips_after": len(refined_knowledge.strips),
            "coverage_score": refined_knowledge.coverage_score,
            "needs_more_info": refined_knowledge.needs_additional_search
        }

        # Log déjà affiché par le refiner

        return refined_knowledge

    def _step_generation(self, query: str, refined_knowledge, steps_details: Dict):
        """
        Étape 5 : Génération de la réponse.
        """
        self._log("\n✍️  ÉTAPE 5 : RESPONSE GENERATION")
        start = time.time()

        # Déterminer le type de question
        question_type = self._detect_question_type(query)

        # Générer la réponse
        response = self.generator.generate_response(
            query=query,
            refined_knowledge=refined_knowledge,
            question_type=question_type
        )

        steps_details["generation"] = {
            "duration": time.time() - start,
            "method": response.generation_method,
            "response_length": len(response.answer),
            "confidence": response.confidence
        }

        self._log(f"   → Méthode : {response.generation_method}")
        self._log(f"   → Longueur : {len(response.answer)} caractères")
        self._log(f"   → Confiance : {response.confidence:.2%}")

        return response

    def _detect_question_type(self, query: str) -> str:
        """
        Détecte le type de question pour adapter la génération.
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["comment", "fonctionne", "marche"]):
            return "comment_fonctionne"
        elif any(word in query_lower for word in ["qu'est-ce", "quoi", "définition"]):
            return "qu_est_ce_que"
        elif "pourquoi" in query_lower:
            return "pourquoi"
        else:
            return "general"

    def _create_debug_info(self, retrieved_docs, graded_strips, refined_knowledge, response):
        """
        Crée des informations de débogage détaillées.
        """
        return {
            "retrieved_docs": [
                {
                    "score": doc["score"],
                    "text_preview": doc["text"][:100] + "...",
                    "metadata": doc.get("metadata", {})
                }
                for doc in retrieved_docs
            ],
            "top_graded_strips": [
                {
                    "score": strip.relevance_score,
                    "category": strip.relevance_category.name,
                    "type": strip.strip.strip_type,
                    "content": strip.strip.content[:100] + "..."
                }
                for strip in graded_strips[:5]
            ],
            "refinement_summary": refined_knowledge.summary,
            "generation_details": response.metadata
        }

    def _log_final_summary(self, result: CRAGResult):
        """
        Affiche un résumé final du traitement.
        """
        self._log(f"\n{'=' * 60}")
        self._log("📊 RÉSUMÉ DU TRAITEMENT")
        self._log(f"{'=' * 60}")
        self._log(f"⏱️  Temps total : {result.processing_time:.2f}s")
        self._log(f"🎯 Confiance : {result.confidence:.2%}")

        if result.needs_more_info:
            self._log("⚠️  Recherche supplémentaire recommandée")
        else:
            self._log("✅ Information suffisante pour répondre")

        # Détail par étape
        self._log("\n📈 Temps par étape :")
        for step, details in result.steps_details.items():
            if "duration" in details:
                self._log(f"   - {step.capitalize()} : {details['duration']:.3f}s")

    def _log(self, message: str, error: bool = False):
        """
        Affiche un message de log si verbose est activé.
        """
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = "❌" if error else "ℹ️"
            print(f"[{timestamp}] {message}")
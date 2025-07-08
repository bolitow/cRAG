from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import requests
import json


@dataclass
class GeneratedResponse:
    """
    Représente la réponse finale générée par le système CRAG.
    """
    answer: str  # La réponse complète
    confidence: float  # Confiance dans la réponse
    sources_used: List[str]  # Strips utilisés pour générer
    generation_method: str  # Méthode utilisée (template, llm, hybrid)
    metadata: Dict  # Métadonnées supplémentaires


class ResponseGenerator:
    """
    Le Response Generator transforme le contexte raffiné en une réponse
    claire, cohérente et complète.

    Il peut utiliser :
    1. Des templates pour structurer la réponse
    2. Un LLM pour générer du texte naturel
    3. Une approche hybride combinant les deux
    """

    def __init__(self,
                 llm_model=None,
                 llm_endpoint: Optional[str] = None,
                 use_templates: bool = True,
                 use_llm: bool = False,
                 response_style: str = "educational"):
        """
        Args:
            llm_model: Modèle de langage local (optionnel)
            llm_endpoint: URL de l'endpoint LLM (format OpenAI)
            use_templates: Utiliser des templates pour structurer
            use_llm: Utiliser le LLM pour générer
            response_style: Style de réponse ("educational", "concise", "detailed")
        """
        self.llm_model = llm_model
        self.llm_endpoint = llm_endpoint
        self.use_templates = use_templates
        self.use_llm = use_llm
        self.response_style = response_style

        # Vérifier la configuration
        if self.use_llm and not self.llm_endpoint and not self.llm_model:
            print("⚠️ LLM activé mais aucun endpoint ou modèle fourni. Fallback sur templates.")
            self.use_llm = False
            self.use_templates = True

        # Initialiser les templates
        self._init_response_templates()

        # Tester la connexion au LLM si endpoint fourni
        if self.llm_endpoint:
            self._test_llm_connection()

    def _test_llm_connection(self):
        """
        Teste la connexion à l'endpoint LLM.
        """
        try:
            test_payload = {
                "messages": [
                    {"role": "user", "content": "Test"}
                ],
                "max_tokens": 10
            }

            response = requests.post(
                self.llm_endpoint,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                print(f"✅ Connexion au LLM établie : {self.llm_endpoint}")
            else:
                print(f"⚠️ LLM accessible mais erreur : {response.status_code}")
        except Exception as e:
            print(f"⚠️ Impossible de se connecter au LLM : {str(e)}")
            print("   Les réponses utiliseront les templates par défaut.")

    def _init_response_templates(self):
        """
        Initialise les templates pour différents types de réponses.
        """
        self.templates = {
            "comment_fonctionne": {
                "educational": """
{definition}

**Comment ça fonctionne :**

{mecanisme}

{details}

{exemple}

En résumé, {resume}
                """,
                "concise": """
{definition_courte}

Fonctionnement : {mecanisme_court}

{exemple_court}
                """,
                "detailed": """
## Définition et Vue d'Ensemble

{definition}

## Mécanisme de Fonctionnement

{mecanisme}

### Détails Techniques

{details}

### Architecture et Composants

{composants}

## Exemples d'Application

{exemple}

## Points Clés à Retenir

{points_cles}
                """
            },
            "general": {
                "educational": """
{definition}

{mecanisme}

{details}

{exemple}
                """,
                "concise": "{definition_courte} {mecanisme_court}",
                "detailed": """
{definition}

{mecanisme}

{details}

{composants}

{exemple}

{points_cles}
                """
            },
            "pourquoi": {
                "educational": """
{definition}

{justification}

{avantages}

{contexte}
                """,
                "concise": "{justification}",
                "detailed": """
## Contexte

{definition}

## Raisons et Justifications

{justification}

## Avantages

{avantages}

## Impact et Applications

{contexte}
                """
            },
            "qu_est_ce_que": {
                "educational": """
{definition}

{caracteristiques}

{contexte}
                """,
                "concise": "{definition_courte}",
                "detailed": """
## Définition

{definition}

## Caractéristiques

{caracteristiques}

## Contexte et Utilisation

{contexte}

## Exemples

{exemples}
                """
            }
        }

        # Templates pour les éléments de réponse
        self.element_templates = {
            "liste": "- {item}\n",
            "numerotation": "{num}. {item}\n",
            "exemple": "Par exemple, {exemple}",
            "definition": "{sujet} est {definition}",
            "mecanisme": "{sujet} fonctionne en {action}"
        }

    def generate_response(self,
                          query: str,
                          refined_knowledge: 'RefinedKnowledge',
                          question_type: str = "comment_fonctionne",
                          max_length: int = 500) -> GeneratedResponse:
        """
        Génère la réponse finale à partir du contexte raffiné.

        Cette méthode orchestre tout le processus de génération.
        """
        # Extraire les éléments du contexte
        context_elements = self._extract_context_elements(refined_knowledge)

        # Déterminer la méthode de génération
        if self.use_llm and (self.llm_endpoint or self.llm_model):
            # Génération par LLM
            response = self._generate_with_llm(
                query,
                refined_knowledge,
                max_length
            )
            method = "llm"
        elif self.use_templates:
            # Génération par templates
            response = self._generate_with_templates(
                question_type,
                context_elements,
                self.response_style
            )
            method = "template"
        else:
            # Approche hybride si les deux sont activés
            response = self._generate_hybrid(
                query,
                refined_knowledge,
                question_type,
                context_elements
            )
            method = "hybrid"

        # Post-traitement
        response = self._post_process_response(response, max_length)

        # Calculer la confiance
        confidence = self._calculate_confidence(refined_knowledge, response)

        # Extraire les sources utilisées
        sources = [strip.strip.content[:50] + "..."
                   for strip in refined_knowledge.strips]

        return GeneratedResponse(
            answer=response,
            confidence=confidence,
            sources_used=sources,
            generation_method=method,
            metadata={
                "question_type": question_type,
                "style": self.response_style,
                "strips_used": len(refined_knowledge.strips),
                "coverage_score": refined_knowledge.coverage_score
            }
        )

    def _extract_context_elements(self,
                                  refined_knowledge: 'RefinedKnowledge') -> Dict[str, str]:
        """
        Extrait les éléments structurés du contexte pour les templates.

        Cette méthode analyse intelligemment les strips pour identifier
        les différentes parties de la réponse.
        """
        elements = {
            "definition": "",
            "definition_courte": "",
            "mecanisme": "",
            "mecanisme_court": "",
            "details": "",
            "composants": "",
            "exemple": "",
            "exemple_court": "",
            "caracteristiques": "",
            "contexte": "",
            "resume": "",
            "points_cles": [],
            "justification": "",
            "avantages": "",
            "exemples": ""
        }

        for strip in refined_knowledge.strips:
            content = strip.strip.content
            strip_type = strip.strip.strip_type

            # Extraire selon le type
            if strip_type == "definition":
                if not elements["definition"]:
                    elements["definition"] = content
                    # Version courte = première phrase
                    elements["definition_courte"] = content.split('.')[0] + "."

            elif "mécanisme" in content.lower() or "utilise" in content.lower():
                if not elements["mecanisme"]:
                    elements["mecanisme"] = content
                    # Version courte
                    elements["mecanisme_court"] = self._summarize_text(content, 50)

            elif strip_type == "enumeration" or "- " in content:
                elements["composants"] = content

            elif any(word in content.lower() for word in ["bert", "gpt", "exemple"]):
                if not elements["exemple"]:
                    elements["exemple"] = content
                    elements["exemple_court"] = content.split('.')[0] + "."

            # Détails techniques
            elif any(word in content.lower() for word in
                     ["permet", "calcule", "traite", "capture"]):
                if elements["details"]:
                    elements["details"] += "\n\n" + content
                else:
                    elements["details"] = content

            # Justifications (pour "pourquoi")
            elif any(word in content.lower() for word in
                     ["révolutionnaire", "transformé", "important", "essentiel"]):
                if not elements["justification"]:
                    elements["justification"] = content

            # Avantages
            elif any(word in content.lower() for word in
                     ["avantage", "permet", "facilite", "améliore", "efficace"]):
                if not elements["avantages"]:
                    elements["avantages"] = content

        # Générer un résumé si on a les éléments
        if elements["definition"] and elements["mecanisme"]:
            sujet = self._extract_subject(elements["definition"])
            elements[
                "resume"] = f"{sujet} utilisent {self._extract_key_mechanism(elements['mecanisme'])} pour accomplir leur tâche"

        # Points clés
        elements["points_cles"] = self._extract_key_points(refined_knowledge)

        return elements

    def _generate_with_templates(self,
                                 question_type: str,
                                 context_elements: Dict[str, str],
                                 style: str) -> str:
        """
        Génère la réponse en utilisant des templates.

        Avantage : Contrôle total sur la structure
        Inconvénient : Moins flexible
        """
        # Mapper les types de questions aux templates disponibles
        template_mapping = {
            "comment_fonctionne": "comment_fonctionne",
            "qu_est_ce_que": "qu_est_ce_que",
            "pourquoi": "pourquoi",
            "general": "general"
        }

        # Obtenir le bon type de template
        template_type = template_mapping.get(question_type, "general")

        # Sélectionner le template approprié
        if template_type in self.templates and style in self.templates[template_type]:
            template = self.templates[template_type][style]
        else:
            # Template par défaut
            template = self.templates["general"]["educational"]

        # Remplir le template
        response = template

        for key, value in context_elements.items():
            placeholder = "{" + key + "}"
            if placeholder in response:
                if isinstance(value, list):
                    # Pour les listes
                    formatted_value = "\n".join([f"- {item}" for item in value]) if value else ""
                    response = response.replace(placeholder, formatted_value)
                else:
                    response = response.replace(placeholder, str(value) if value else "")

        # Nettoyer les placeholders non remplis
        response = re.sub(r'\{[^}]+\}', '', response)

        # Nettoyer les espaces multiples et lignes vides
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'^\s*\n', '', response, flags=re.MULTILINE)
        response = response.strip()

        return response

    def _generate_with_llm(self,
                           query: str,
                           refined_knowledge: 'RefinedKnowledge',
                           max_length: int) -> str:
        """
        Génère la réponse en utilisant un LLM via API ou local.

        Supporte les endpoints compatibles OpenAI.
        """
        # Construire le contexte à partir des strips
        context = "\n\n".join([strip.strip.content for strip in refined_knowledge.strips])

        # Créer le prompt selon le style
        if self.response_style == "educational":
            system_prompt = """Tu es un assistant pédagogique qui donne des réponses claires et structurées.
Utilise le contexte fourni pour répondre de manière précise et éducative.
Structure ta réponse avec des sections si nécessaire."""
        elif self.response_style == "concise":
            system_prompt = """Tu es un assistant qui donne des réponses brèves et directes.
Réponds de manière concise en utilisant uniquement les informations du contexte."""
        else:  # detailed
            system_prompt = """Tu es un assistant expert qui fournit des réponses détaillées et complètes.
Utilise tout le contexte disponible pour donner une réponse approfondie avec des exemples."""

        user_prompt = f"""Réponds à cette question en utilisant UNIQUEMENT les informations du contexte fourni.
Ne fais pas de suppositions ou d'ajouts non présents dans le contexte.

Question : {query}

Contexte :
{context}

Réponse :"""

        # Si on a un endpoint HTTP
        if self.llm_endpoint:
            try:
                # Préparer la requête au format OpenAI
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_length,
                    "temperature": 0.7,
                    "stream": False
                }

                # Faire la requête
                response = requests.post(
                    self.llm_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    # Extraire la réponse selon le format de retour
                    if "choices" in result and len(result["choices"]) > 0:
                        # Format OpenAI standard
                        return result["choices"][0]["message"]["content"].strip()
                    elif "response" in result:
                        # Format alternatif possible
                        return result["response"].strip()
                    elif "text" in result:
                        # Autre format possible
                        return result["text"].strip()
                    else:
                        # Essayer de récupérer n'importe quelle string dans la réponse
                        return str(result).strip()
                else:
                    print(f"⚠️ Erreur LLM API: {response.status_code} - {response.text}")
                    return self._fallback_to_template_response(query, context)

            except requests.exceptions.Timeout:
                print("⚠️ Timeout de l'API LLM. Fallback sur template.")
                return self._fallback_to_template_response(query, context)
            except Exception as e:
                print(f"⚠️ Erreur lors de l'appel LLM: {str(e)}. Fallback sur template.")
                return self._fallback_to_template_response(query, context)

        # Si on a un modèle local
        elif self.llm_model:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.llm_model(prompt, max_length=max_length)
            return response

        # Fallback
        return self._fallback_to_template_response(query, context)

    def _fallback_to_template_response(self, query: str, context: str) -> str:
        """
        Réponse de secours si le LLM n'est pas disponible.
        """
        # Extraire les premières phrases du contexte
        sentences = context.split('.')[:3]
        summary = '. '.join(sentences) + '.'

        return f"""D'après les informations disponibles :

{summary}

[Note: Réponse générée par template car le LLM n'est pas disponible]"""

    def _generate_hybrid(self,
                         query: str,
                         refined_knowledge: 'RefinedKnowledge',
                         question_type: str,
                         context_elements: Dict[str, str]) -> str:
        """
        Approche hybride : structure par template, enrichissement par LLM.
        """
        # Générer la structure de base avec template
        base_response = self._generate_with_templates(
            question_type,
            context_elements,
            "educational"
        )

        # Si on a un LLM disponible, l'utiliser pour enrichir
        if self.llm_endpoint or self.llm_model:
            if self.llm_endpoint:
                try:
                    enrichment_prompt = f"""Améliore cette réponse en la rendant plus fluide et naturelle.
Garde tous les faits mais améliore le style et la clarté.

Réponse originale :
{base_response}

Version améliorée :"""

                    payload = {
                        "messages": [
                            {"role": "system",
                             "content": "Tu es un assistant qui améliore la qualité rédactionnelle des textes."},
                            {"role": "user", "content": enrichment_prompt}
                        ],
                        "max_tokens": len(base_response) * 2,
                        "temperature": 0.5
                    }

                    response = requests.post(
                        self.llm_endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"].strip()
                except:
                    pass  # Si erreur, on garde la version template

            elif self.llm_model:
                enrichment_prompt = f"""
Améliore cette réponse en la rendant plus fluide et naturelle,
sans changer les informations factuelles :

{base_response}

Version améliorée :
"""
                enhanced = self.llm_model(enrichment_prompt, max_length=len(base_response) * 2)
                return enhanced

        return base_response

    def _post_process_response(self, response: str, max_length: int) -> str:
        """
        Post-traitement pour améliorer la qualité de la réponse.
        """
        # Nettoyer les espaces
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)

        # Assurer une ponctuation correcte
        if response and not response[-1] in '.!?':
            response += '.'

        # Tronquer si trop long
        if len(response) > max_length:
            # Couper à la dernière phrase complète
            sentences = response.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) + 1 <= max_length:
                    truncated += sentence + "."
                else:
                    break
            response = truncated.strip()

        return response

    def _calculate_confidence(self,
                              refined_knowledge: 'RefinedKnowledge',
                              response: str) -> float:
        """
        Calcule la confiance dans la réponse générée.
        """
        factors = []

        # Facteur 1 : Couverture de la question
        factors.append(refined_knowledge.coverage_score)

        # Facteur 2 : Nombre de strips utilisés
        strips_score = min(len(refined_knowledge.strips) / 5, 1.0)
        factors.append(strips_score)

        # Facteur 3 : Longueur de la réponse (ni trop courte, ni trop longue)
        ideal_length = 300
        length_score = 1 - abs(len(response) - ideal_length) / ideal_length
        length_score = max(0, min(1, length_score))
        factors.append(length_score)

        # Facteur 4 : Qualité moyenne des strips
        if refined_knowledge.strips:
            avg_quality = sum(s.relevance_score for s in refined_knowledge.strips) / len(refined_knowledge.strips)
            factors.append(avg_quality)

        return sum(factors) / len(factors) if factors else 0.5

    def _extract_subject(self, definition: str) -> str:
        """
        Extrait le sujet principal d'une définition.
        """
        # Chercher "X est/sont"
        match = re.search(r'([\w\s]+)\s+(?:est|sont)', definition)
        if match:
            return match.group(1).strip()

        # Sinon, prendre les premiers mots
        words = definition.split()[:3]
        return " ".join(words)

    def _extract_key_mechanism(self, mechanism_text: str) -> str:
        """
        Extrait le mécanisme clé d'un texte.
        """
        # Chercher après "utilise", "emploie", etc.
        patterns = [
            r'utilise[nt]?\s+([^.]+)',
            r'emploie[nt]?\s+([^.]+)',
            r'mécanisme\s+(?:de\s+)?([^.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, mechanism_text.lower())
            if match:
                return match.group(1).strip()

        # Fallback : prendre une partie du texte
        return "leur mécanisme spécifique"

    def _extract_key_points(self, refined_knowledge: 'RefinedKnowledge') -> List[str]:
        """
        Extrait les points clés de la connaissance raffinée.
        """
        key_points = []

        # Chercher les affirmations importantes dans les strips
        for strip in refined_knowledge.strips[:5]:  # Top 5 strips
            content = strip.strip.content

            # Chercher les phrases clés
            if any(word in content.lower() for word in ["clé", "principal", "important"]):
                # Extraire la phrase
                sentences = content.split('.')
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ["clé", "principal"]):
                        key_points.append(sentence.strip() + ".")
                        break

        # Limiter à 3 points
        return key_points[:3]

    def _summarize_text(self, text: str, max_words: int) -> str:
        """
        Résume un texte en gardant les mots les plus importants.
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        # Garder le début et ajouter "..."
        return " ".join(words[:max_words]) + "..."
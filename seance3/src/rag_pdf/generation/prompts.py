"""
Templates de prompts versionnés.

Les prompts sont du code à part entière : ils méritent leur fichier,
leur revue, et leur historique git. Les "garder dans un notebook"
est un anti-pattern qui se paie en production.

Trois prompts ici :
1. RAG_SYSTEM_PROMPT — pour la génération de réponse (séance 2)
2. QUERY_REWRITE_PROMPT — pour reformuler la question (séance 3, pré-retrieval)
3. HYDE_PROMPT — pour générer une réponse hypothétique (séance 3, bonus)
"""
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------------
# 1. Prompt de génération de la réponse finale (RAG)
# ----------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """Tu es un assistant pédagogique qui répond aux questions \
en t'appuyant UNIQUEMENT sur le contexte fourni ci-dessous.

Règles strictes (à respecter à la lettre) :
1. Si la réponse ne se trouve pas dans le contexte, dis-le clairement :
   « Je n'ai pas trouvé cette information dans le document. »
2. N'invente JAMAIS d'information qui ne soit pas dans le contexte.
3. Cite systématiquement les pages sources entre crochets : [Page X].
   Si plusieurs pages sont concernées : [Page X, Page Y].
4. Réponds en français, de manière claire, structurée et concise.

CONTEXTE :
{context}
"""

RAG_USER_PROMPT = """QUESTION : {question}"""


def build_rag_prompt() -> ChatPromptTemplate:
    """Construit le template de prompt utilisé par la chaîne RAG."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ]
    )


# ----------------------------------------------------------------------
# 2. Prompt de reformulation de question (pré-retrieval)
# ----------------------------------------------------------------------

QUERY_REWRITE_SYSTEM = """Tu es un assistant spécialisé dans la reformulation de \
questions pour un moteur de recherche sémantique.

Ton objectif : reformuler la question de l'utilisateur pour qu'elle soit \
plus précise, plus explicite, et donc plus facile à matcher avec un corpus documentaire.

Règles :
1. Garde le SENS de la question d'origine, ne change pas le sujet.
2. Explicite les références vagues ("ce truc", "ça", "comment ça marche").
3. Ajoute des mots-clés techniques si tu en identifies dans le contexte du sujet.
4. Reste concis : 1 à 2 phrases maximum.
5. Réponds UNIQUEMENT par la question reformulée, sans explication ni préambule.

Exemple :
  Question d'origine : « Comment ça marche au fait ? »
  Question reformulée : « Comment fonctionne un système RAG : quelles sont les étapes du pipeline d'indexation et de récupération ? »
"""

QUERY_REWRITE_USER = """Question d'origine : {question}

Question reformulée :"""


def build_query_rewrite_prompt() -> ChatPromptTemplate:
    """Prompt pour reformuler une question utilisateur."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_REWRITE_SYSTEM),
            ("human", QUERY_REWRITE_USER),
        ]
    )


# ----------------------------------------------------------------------
# 3. Prompt HyDE (Hypothetical Document Embeddings) — bonus séance 3
# ----------------------------------------------------------------------

HYDE_SYSTEM = """Tu es un assistant qui génère une RÉPONSE PLAUSIBLE à une question, \
sans contexte réel.

Cette réponse sera utilisée pour la recherche sémantique : on embedde ta \
réponse hypothétique, et on cherche les chunks réels qui lui ressemblent. \
Contre-intuitif mais souvent meilleur que d'embedder la question seule, \
car les réponses se ressemblent plus aux chunks que les questions.

Règles :
1. Génère un paragraphe court (3-5 phrases) qui ressemble à une réponse plausible.
2. Utilise le vocabulaire et le style d'un texte technique sur le sujet.
3. Tu PEUX inventer — le but n'est pas d'être exact, mais de ressembler à un vrai passage.
4. Réponds UNIQUEMENT par le paragraphe, sans préambule.
"""

HYDE_USER = """Question : {question}

Réponse hypothétique :"""


def build_hyde_prompt() -> ChatPromptTemplate:
    """Prompt pour générer une réponse hypothétique (HyDE)."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", HYDE_SYSTEM),
            ("human", HYDE_USER),
        ]
    )

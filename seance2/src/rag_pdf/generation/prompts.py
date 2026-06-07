"""
Templates de prompts versionnés.

Les prompts sont du code à part entière : ils méritent leur fichier,
leur revue, et leur historique git. Les "garder dans un notebook"
est un anti-pattern qui se paie en production.

Le prompt système ici applique trois bonnes pratiques cruciales pour
un RAG :
1. Restriction explicite au contexte fourni → anti-hallucination.
2. Aveu d'ignorance autorisé ("Je n'ai pas trouvé…") → essentiel.
3. Obligation de citer les pages → traçabilité.
"""
from langchain_core.prompts import ChatPromptTemplate


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

"""
Factory : construit le pipeline RAG complet à partir de la config.

C'est ici qu'on assemble tous les composants. Un seul `build_rag_chain()`
suffit pour récupérer un pipeline fonctionnel : c'est l'API publique
qu'utilisera tout le reste du projet (scripts, notebooks, app Streamlit).

Avantage : si on change un composant (ex : passage à Pinecone), on ne
modifie que cette factory — pas le code appelant.
"""
from rag_pdf.config import settings
from rag_pdf.generation.llm import GroqLLM
from rag_pdf.indexing.embedder import HFEmbedder
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.pipeline.rag_chain import RAGChain
from rag_pdf.retrieval.retriever import Retriever


def build_rag_chain() -> RAGChain:
    """Construit un pipeline RAG complet depuis la configuration centralisée.

    Returns:
        Une instance de `RAGChain` prête à être interrogée via `.invoke(question)`.

    Préconditions :
        Une base Chroma doit déjà avoir été remplie via le script d'ingestion :
            `python -m scripts.ingest data/pdfs/<votre_pdf>.pdf`
    """
    embedder = HFEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    vector_store = ChromaVectorStore(
        embedder=embedder,
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )
    retriever = Retriever(vector_store=vector_store, k=settings.top_k)
    llm = GroqLLM(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )
    return RAGChain(retriever=retriever, llm=llm)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(k_faq=3, k_tickets=3, k_guides=3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    def retrieve(query: str) -> list[Document]:
        faq_results = faq_store.similarity_search_with_score(query, k=k_faq)
        ticket_results = tickets_store.similarity_search_with_score(query, k=k_tickets)
        guide_results = guides_store.similarity_search_with_score(query, k=k_guides)

        all_results = faq_results + ticket_results + guide_results

        best_score = min(score for doc, score in all_results)

        if best_score > 1.0:
            return []

        return [doc for doc, score in all_results]

    return RunnableLambda(retrieve)
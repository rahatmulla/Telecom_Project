"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)                                     #no chunks as it is already in small pieces
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda                                 #lamda turns Python function to be used inside a LangChain pipeline
from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_retriever(
    k_faq: int = 3,                                                     #k is the number of top documents to retrieve from each collection
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})                              # how many results to retrieve
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})

    #creates one function that searches all 3 knowledge bases.
    def retrieve(query: str) -> list[Document]:                                     #takes user question
        return (
            faq_retriever.invoke(query)                                                #searches FAQ
            + tickets_retriever.invoke(query)                                           #searches tickets
            + guides_retriever.invoke(query)                                            #searches guides
        )

    return RunnableLambda(retrieve)                                                 #lambda lets langchain use the retrieve function

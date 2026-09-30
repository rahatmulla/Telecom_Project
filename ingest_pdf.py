"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"                      #to print only error messages from BTS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH   = os.path.join("data", "telecom_guide.pdf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()                               #pages is a list of Document objects, each representing a page in the PDF
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],            #tries paragraphs first, then lines, then sentences, then words
    )
    chunks = splitter.split_documents(pages)                #document object ->representing chunks of text from the PDF

    # Labelling the chunks with metadata and index
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")                      

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)                      #loading the embedding model from HuggingFace

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(                                                    #creating chromadb vectorstore 
        documents=chunks,   
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,                                                       #storing data locally, eliminates need to recreate embeddings everytime
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")                     #prints no of vectors stored


if __name__ == "__main__":
    main()

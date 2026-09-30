"""
Ingests resolved tickets from data/tickets.db into the 'tickets' Chroma collection.
Run once (or after adding new tickets): python ingest_tickets.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sqlite3
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION  = "tickets"
DB_PATH     = os.path.join("data", "tickets.db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_ticket_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)                                                     #conencts to the database
    conn.row_factory = sqlite3.Row
    rows = conn.execute(                                                               #get resolved tickets from the db
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()
    conn.close()                                                                       #end connection to the db

    docs = []                                                                               #holds ticktet documents
    for row in rows:                                                                        #goes through each row 
        # Combine issue description + resolution into a single searchable text block
        content = (                                                     
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(                                                                   #creates lang chain document containing that text
            page_content=content,
            metadata={
                "source":    "ticket",
                "ticket_id": row["ticket_id"],
                "category":  row["category"],
                "status":    row["status"],
            },
        ))
    return docs

def main():
    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(DB_PATH)
    print(f"  {len(docs)} resolved tickets loaded.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)                              #load hugging face model

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(                                                            #inserts the docs in chromaDB
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,                                                           #store vectors locally
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")                             #returns vector count in the collection


if __name__ == "__main__":
    main()
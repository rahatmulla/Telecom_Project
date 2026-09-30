# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by retrieving relevant context from three knowledge sources and generating responses with Qwen via Groq.

## Knowledge Base

- **SQLite Database:** Past resolved support tickets containing issue type, description, resolution, status, etc.
- **CSV File:** Frequently asked questions and their corresponding answers.
- **PDF Document:** Telecom Technical Reference Guide for Customer Care & Network Operations.

## Technologies

- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Library:** Hugging Face Sentence Transformers
- **LLM:** Qwen via Groq
- **Web Application:** Streamlit

## How It Works

### 1. Indexing

The knowledge sources are processed and stored in ChromaDB.

- **PDF:** Split into smaller chunks of 600 characters with 100-character overlap.
- **SQLite:** Each resolved ticket is treated as one document since the records are already small.
- **FAQ CSV:** Each FAQ row is treated as one document.

A Sentence Transformer model from Hugging Face converts each document into a numerical embedding.

The embeddings, text, and metadata are then stored in separate ChromaDB collections:

- `faq` — FAQ entries
- `tickets` — Resolved support tickets
- `guides` — PDF guide chunks

### 2. Retrieval

When a user asks a question:

1. The user's question is converted into an embedding using the same Sentence Transformer model.
2. Semantic search compares the question embedding with the embeddings stored in ChromaDB.
3. A merged retriever searches all three collections.
4. The top 3 relevant results are retrieved from each collection, providing up to 9 relevant documents.
5. The retrieved documents are combined with the user's question.
6. `ChatPromptTemplate` formats the question and retrieved context.
7. The prompt is sent to Qwen via Groq.
8. The LLM generates the final answer.

## Architecture

<img width="544" height="1404" alt="Telegram Chatbot Architecture drawio (1)" src="https://github.com/user-attachments/assets/876fad96-e941-4467-a9f1-087b98671e68" />



## Data Sources

| Collection | Source | Granularity |
|------------|--------|-------------|
| FAQ | `data/faq.csv` | 1 document per FAQ row |
| Tickets | `data/tickets.db` | 1 document per resolved ticket |
| Guides | `data/telecom_guide.pdf` | 600-character chunks with 100-character overlap | 
  
## Project Structure  

```
rag-telecom-chatbot/
│
├── app.py                 # Streamlit web interface
├── main.py                # CLI entry point
├── rag_chain.py           # RAG chain and LLM configuration
├── retriever.py           # Merges the three ChromaDB retrievers
│
├── ingest_faq.py          # Ingests FAQ data
├── ingest_tickets.py      # Ingests resolved tickets
├── ingest_pdf.py          # Ingests and chunks PDF
│
├── data/
│   ├── faq.csv
│   ├── tickets.db
│   ├── telecom_guide.pdf
│   ├── seed_tickets.py
│   └── generate_pdf.py
│
├── chroma_store/          # Persisted ChromaDB data
├── pyproject.toml
├── uv.lock
└── .env.example
```

  
  
## RAG Retrieval Confidence Thresholding:

** Confidence & Fallback Logic **

The system uses 3 Chroma collections:
FAQ
Tickets
Guides

A temporary test was performed using similarity_search_with_score() to measure how closely retrieved documents match the user's question.
Both relevant telecom questions and irrelevant questions were tested.

The results showed:
Relevant questions → scores mostly below 1.0
Irrelevant questions → scores mostly above 1.5
Based on these initial tests, a starting threshold of 1.0 was selected.

The intended fallback flow is:

Question
   ↓
Search Chroma
   ↓
Best score ≤ 1.0?
   ↓
YES → Send to LLM
NO  → "I don't know, please call 611"

Project 2 Exercise: Telecom RAG Improvements

The chatbot was further enhanced with source citations, confidence-based fallback logic, and retrieval evaluation.

1. Source Citations

The document formatting and system prompt were modified so that generated answers explicitly identify the source of retrieved information.

Depending on the source, the chatbot can cite:

FAQ: FAQ source
Support Ticket: Ticket ID
Technical Guide: Guide page number

This makes the chatbot's answers more traceable and allows the user to understand where the retrieved information came from.

2. Confidence / Fallback Logic

The retrieval process was modified to use similarity_search_with_score() instead of relying only on a standard retriever.

The system checks the similarity score of the retrieved results against the configured threshold.

User Question
      ↓
Retrieve documents with scores
      ↓
Check best similarity score
      ↓
Best score ≤ 1.0?
   ┌───────┴───────┐
  YES              NO
   ↓                ↓
Send context      Skip LLM
to LLM              ↓
   ↓          "I don't know,
Generate answer  please call 611"

If no retrieved result meets the threshold:

The LLM is not called.
The chatbot returns the predefined fallback response.
This reduces the risk of generating unsupported answers from weakly related context.
3. Retrieval Quality Evaluation

A retrieval evaluation script was created to measure retrieval performance.

The evaluation uses 10 hand-crafted question and expected-ticket ID pairs.

For each test question:

The question is sent to the ticket retriever.
The top 3 results are retrieved.
The returned ticket IDs are compared with the expected ticket ID.
The result is counted as successful if the expected ticket appears within the top 3.

The evaluation measures Top-3 Recall:

Top-3 Recall =
Number of questions where the expected ticket
appears in the top 3 results
/
Total number of test questions

This provides a simple quantitative way to evaluate retrieval quality rather than relying only on manual testing.

Key RAG Features
Multi-source retrieval across FAQ, support tickets, and technical documentation
Separate ChromaDB collections for each knowledge source
Semantic search using Sentence Transformers
Top-k retrieval across multiple sources
Source-aware citations
Similarity-score-based confidence checking
LLM fallback prevention for low-confidence retrievals
Automated Top-3 retrieval evaluation
Streamlit-based chatbot interface
Qwen LLM integration through Groq
Environment Variables








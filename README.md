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

```text
                         User Question
                              │
                              ▼
                    Question Embedding
                              │
                              ▼
                     Merged Retriever
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        ChromaDB          ChromaDB         ChromaDB
           FAQ             Tickets           Guides
             │                │                │
          Top 3             Top 3             Top 3
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                     Relevant Documents
                         (up to 9)
                              │
                              ▼
                    ChatPromptTemplate
                  Question + Retrieved Context
                              │
                              ▼
                       Qwen via Groq
                              │
                              ▼
                            Answer
Data Sources
Collection	Source	Granularity
faq	data/faq.csv	1 document per FAQ row
tickets	data/tickets.db	1 document per resolved ticket
guides	data/telecom_guide.pdf	600-character chunks with 100-character overlap
Project Structure
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
Setup
Prerequisites
Python 3.11+
Groq API key
Hugging Face token
Install
uv sync

Or:

pip install -e .
Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
Build the Vector Database

Run the ingestion scripts:

python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py

These scripts process the source data, generate embeddings, and store them in ChromaDB.

Run the Application

Streamlit:

streamlit run app.py

CLI:

python main.py
Regenerating Data

If the source data changes, regenerate and re-ingest the corresponding source.

python data/seed_tickets.py
python data/generate_pdf.py

Then run the appropriate ingestion script again.


### One thing I would change before pushing to GitHub

Do **not** upload:

```text
.env
chroma_store/

Your .gitignore should contain:

.env
chroma_store/
__pycache__/
*.pyc
.venv/

Keep .env.example in GitHub, but never your actual API keys.
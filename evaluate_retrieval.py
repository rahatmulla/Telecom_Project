from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

tickets_store = Chroma(
    collection_name="tickets",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

test_cases = [
    ("Why is my internet not working?", "TK-001"),
    ("Why does my signal keep dropping?", "TK-002"),
    ("Why is my data balance not updating?", "TK-003"),
    ("Why did I get unexpected roaming charges?", "TK-004"),
    ("Why is my SIM not recognised?", "TK-005"),
    ("Why was I charged twice for my monthly plan?", "TK-006"),
    ("Why are my calls going straight to voicemail?", "TK-007"),
    ("Why is my 4G internet extremely slow?", "TK-008"),
    ("Why is my eSIM activation failing?", "TK-009"),
    ("Why can't I view or download my itemised bill?", "TK-010"),
]

correct = 0

for question, expected_id in test_cases:

    results = tickets_store.similarity_search(question, k=3)

    retrieved_ids = [
        doc.metadata["ticket_id"]
        for doc in results
    ]

    if expected_id in retrieved_ids:
        correct += 1
        print("PASS:", question)
    else:
        print("FAIL:", question)
        print("Retrieved:", retrieved_ids)

recall = correct / len(test_cases)

print(f"\nTop-3 Recall: {recall:.0%}")
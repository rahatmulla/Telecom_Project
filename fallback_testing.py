from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# Load the existing collections
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


# Questions we want to test.
# Includes both questions system SHOULD answer
# and questions it SHOULD NOT answer.
questions = [

    # ─────────────────────────────────────────────
    # 10 KNOWN — SHOULD ANSWER
    # ─────────────────────────────────────────────

    "When is my monthly bill due?",
    "Why is my bill higher than usual?",
    "How do I set up AutoPay?",
    "How can I download my itemised bill?",
    "How do I unlock my phone to use another network?",
    "I purchased a 5 GB add-on but my data balance has not updated.",
    "I was charged twice because of a payment gateway retry. What should I do?",
    "What is VoLTE and what benefit does it provide?",
    "What is a Call Detail Record and how is it used for billing?",
    "What information must an agent verify before sharing account details?",


    # ─────────────────────────────────────────────
    # 10 UNKNOWN — SHOULD NOT ANSWER, SHOULD REFER TO CALLING 611 OR USING THE APP
    # ─────────────────────────────────────────────

    "What is the weather today?",
    "How do I cook biryani?",
    "Who won the football match yesterday?",
    "What is the capital of France?",
    "How do I apply for a driving licence?",
    "What are the best restaurants in Dubai?",
    "How do I repair a leaking kitchen tap?",
    "What is the current price of gold?",
    "How do I learn Python programming?",
    "What are the best universities in the UAE?",
]


# Test every question
for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    # Test the question against each of the three Chroma collections
    for name, store in [
        ("FAQ", faq_store),
        ("TICKETS", tickets_store),
        ("GUIDES", guides_store),
    ]:

        # Retrieve the 3 most similar documents AND their similarity scores
        results = store.similarity_search_with_score(question, k=3)

        print(f"\n{name}:")

        # Display the score and a small part of each retrieved document
        for doc, score in results:
            print("Score:", round(score, 4))
            print(
                "Text:",
                doc.page_content[:150].replace("\n", " ")
            )
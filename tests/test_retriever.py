import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import build_retriever


def test_retriever_returns_results_for_known_question():
    retriever = build_retriever()
    results = retriever.invoke("How do I set up AutoPay?")
    assert len(results) > 0


def test_retriever_returns_empty_for_irrelevant_question():
    retriever = build_retriever()
    results = retriever.invoke("What is the capital of France?")
    assert results == []
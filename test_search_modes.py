
import numpy as np

from search_engine import search_chunks
from semantic_search import semantic_search
from answer_generator import build_grounded_response


class FakeEmbeddingModel:
    """Simulated embedding model for repeatable integration tests."""

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            text = text.lower()

            if any(term in text for term in [
                "quantum",
                "spacecraft",
                "telescope",
            ]):
                vectors.append([0.0, 0.0, 0.0, 1.0])

            elif any(term in text for term in [
                "vacation",
                "leave",
                "time off",
                "time away",
                "paid days",
            ]):
                vectors.append([1.0, 0.0, 0.0, 0.0])

            elif any(term in text for term in [
                "refund",
                "billing",
                "payment",
            ]):
                vectors.append([0.0, 1.0, 0.0, 0.0])

            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])

        return np.array(vectors, dtype=float)


SAMPLE_CHUNKS = [
    {
        "source": "employee_handbook.pdf",
        "page": 3,
        "chunk_id": 1,
        "text": (
            "Full-time employees receive 15 paid vacation "
            "days each calendar year."
        ),
    },
    {
        "source": "support_policy.pdf",
        "page": 5,
        "chunk_id": 2,
        "text": (
            "Customers may request refunds within 30 days "
            "of their purchase."
        ),
    },
    {
        "source": "general_guide.txt",
        "page": None,
        "chunk_id": 3,
        "text": (
            "The company provides general information "
            "about its services."
        ),
    },
]


def test_keyword_search_retrieves_vacation_policy():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation days",
    )

    assert len(results) > 0
    assert results[0]["source"] == "employee_handbook.pdf"


def test_semantic_search_retrieves_paraphrased_question():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How much time away from work can employees take?",
        model=FakeEmbeddingModel(),
    )

    assert len(results) > 0
    assert results[0]["source"] == "employee_handbook.pdf"


def test_both_search_methods_produce_source_backed_responses():
    keyword_results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation days",
    )

    semantic_results = semantic_search(
        SAMPLE_CHUNKS,
        "How much time away from work can employees take?",
        model=FakeEmbeddingModel(),
    )

    keyword_response = build_grounded_response(keyword_results)
    semantic_response = build_grounded_response(semantic_results)

    assert keyword_response["status"] == "found"
    assert semantic_response["status"] == "found"

    assert keyword_response["sources"][0]["source"] == (
        "employee_handbook.pdf"
    )

    assert semantic_response["sources"][0]["source"] == (
        "employee_handbook.pdf"
    )

    assert "15 paid vacation days" in keyword_response["answer"]
    assert "15 paid vacation days" in semantic_response["answer"]


def test_semantic_search_rejects_unrelated_content():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "quantum spacecraft telescope",
        model=FakeEmbeddingModel(),
    )

    response = build_grounded_response(results)

    assert results == []
    assert response["status"] == "not_found"
    assert response["sources"] == []

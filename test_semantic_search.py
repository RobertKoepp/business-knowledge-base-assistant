
import numpy as np
import pytest

from semantic_search import semantic_search


class FakeEmbeddingModel:
    """A lightweight model substitute for automated testing."""

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

            # Give unrelated queries their own distinct representation.
            if any(term in text for term in [
                "quantum",
                "spacecraft",
                "telescope",
            ]):
                vectors.append([0.0, 0.0, 0.0, 1.0])

            elif any(term in text for term in [
                "time off",
                "vacation",
                "leave",
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
        "text": "Employees can request paid leave.",
    },
    {
        "source": "support_policy.pdf",
        "page": 5,
        "chunk_id": 2,
        "text": "Customers may request refunds after purchase.",
    },
    {
        "source": "general_guide.txt",
        "page": None,
        "chunk_id": 3,
        "text": "The company provides general information.",
    },
]


def test_semantic_search_ranks_conceptually_related_passage():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How do I request time off?",
        model=FakeEmbeddingModel(),
    )

    assert len(results) > 0
    assert results[0]["source"] == "employee_handbook.pdf"


def test_semantic_search_preserves_source_information():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How do I request time off?",
        model=FakeEmbeddingModel(),
    )

    assert results[0]["source"] == "employee_handbook.pdf"
    assert results[0]["page"] == 3
    assert results[0]["chunk_id"] == 1


def test_semantic_search_returns_similarity_score():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How do I request time off?",
        model=FakeEmbeddingModel(),
    )

    assert "semantic_score" in results[0]
    assert "relevance_score" in results[0]
    assert results[0]["semantic_score"] == 1.0


def test_top_k_limits_results():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "vacation policy",
        top_k=1,
        model=FakeEmbeddingModel(),
    )

    assert len(results) == 1


def test_minimum_score_filters_irrelevant_passages():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How do I request time off?",
        min_score=0.95,
        model=FakeEmbeddingModel(),
    )

    assert len(results) == 1
    assert results[0]["source"] == "employee_handbook.pdf"


def test_empty_chunk_list_returns_no_results():
    results = semantic_search(
        [],
        "vacation policy",
        model=FakeEmbeddingModel(),
    )

    assert results == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="search question"):
        semantic_search(
            SAMPLE_CHUNKS,
            "",
            model=FakeEmbeddingModel(),
        )


def test_non_string_query_is_rejected():
    with pytest.raises(ValueError, match="search question"):
        semantic_search(
            SAMPLE_CHUNKS,
            None,
            model=FakeEmbeddingModel(),
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    [0, -1, 1.5, True],
)
def test_invalid_top_k_is_rejected(invalid_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        semantic_search(
            SAMPLE_CHUNKS,
            "vacation policy",
            top_k=invalid_top_k,
            model=FakeEmbeddingModel(),
        )


@pytest.mark.parametrize(
    "invalid_score",
    [2, -2, float("nan"), True, "invalid"],
)
def test_invalid_minimum_score_is_rejected(invalid_score):
    with pytest.raises(ValueError, match="between -1 and 1"):
        semantic_search(
            SAMPLE_CHUNKS,
            "vacation policy",
            min_score=invalid_score,
            model=FakeEmbeddingModel(),
        )


def test_invalid_document_chunks_are_ignored():
    chunks = [
        None,
        {"source": "empty.txt", "text": "   "},
        {"source": "invalid.txt", "text": 123},
        SAMPLE_CHUNKS[0],
    ]

    results = semantic_search(
        chunks,
        "How do I request time off?",
        model=FakeEmbeddingModel(),
    )

    assert len(results) == 1
    assert results[0]["source"] == "employee_handbook.pdf"


def test_unrelated_query_returns_no_results():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "quantum spacecraft telescope",
        model=FakeEmbeddingModel(),
    )

    assert results == []


def test_results_are_ranked_by_similarity():
    results = semantic_search(
        SAMPLE_CHUNKS,
        "How do I request time off?",
        model=FakeEmbeddingModel(),
    )

    scores = [
        result["semantic_score"]
        for result in results
    ]

    assert scores == sorted(scores, reverse=True)

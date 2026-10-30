
import pytest

from search_engine import search_chunks


SAMPLE_CHUNKS = [
    {
        "source": "employee_handbook.pdf",
        "page": 3,
        "chunk_id": 1,
        "text": (
            "Employees must submit vacation requests at least "
            "thirty days before planned leave."
        ),
    },
    {
        "source": "support_guide.pdf",
        "page": 2,
        "chunk_id": 2,
        "text": (
            "Customers may request refunds within thirty days "
            "of their purchase."
        ),
    },
    {
        "source": "product_manual.docx",
        "page": None,
        "chunk_id": 3,
        "text": (
            "Users can reset their password from the account "
            "settings page."
        ),
    },
]


def test_retrieves_relevant_passage():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation requests planned leave",
    )

    assert len(results) > 0
    assert results[0]["source"] == "employee_handbook.pdf"


def test_most_relevant_passage_is_ranked_first():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation requests",
    )

    assert results[0]["source"] == "employee_handbook.pdf"
    assert "vacation requests" in results[0]["text"]


def test_top_k_limits_results():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "thirty days",
        top_k=1,
    )

    assert len(results) == 1


def test_preserves_source_and_page_information():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation requests",
    )

    assert results[0]["source"] == "employee_handbook.pdf"
    assert results[0]["page"] == 3
    assert results[0]["chunk_id"] == 1


def test_returns_relevance_score():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "vacation requests",
    )

    assert "relevance_score" in results[0]
    assert 0 < results[0]["relevance_score"] <= 1


def test_empty_chunk_list_returns_no_results():
    results = search_chunks(
        [],
        "vacation policy",
    )

    assert results == []


def test_unmatched_query_returns_no_results():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "quantum spacecraft telescope",
    )

    assert results == []


def test_query_with_only_stop_words_returns_no_results():
    results = search_chunks(
        SAMPLE_CHUNKS,
        "the and is",
    )

    assert results == []


def test_documents_without_searchable_words_return_no_results():
    chunks = [
        {
            "source": "empty_content.txt",
            "page": None,
            "chunk_id": 1,
            "text": "the and is or",
        }
    ]

    results = search_chunks(
        chunks,
        "vacation policy",
    )

    assert results == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="search question"):
        search_chunks(SAMPLE_CHUNKS, "")

    with pytest.raises(ValueError, match="search question"):
        search_chunks(SAMPLE_CHUNKS, "   ")


def test_non_string_query_is_rejected():
    with pytest.raises(ValueError, match="search question"):
        search_chunks(SAMPLE_CHUNKS, None)


@pytest.mark.parametrize(
    "invalid_top_k",
    [0, -1, 1.5, True],
)
def test_invalid_top_k_is_rejected(invalid_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        search_chunks(
            SAMPLE_CHUNKS,
            "vacation requests",
            top_k=invalid_top_k,
        )


def test_ignores_invalid_document_chunks():
    chunks = [
        None,
        {"source": "empty.txt", "text": "   "},
        {"source": "invalid.txt", "text": 123},
        SAMPLE_CHUNKS[0],
    ]

    results = search_chunks(
        chunks,
        "vacation requests",
    )

    assert len(results) == 1
    assert results[0]["source"] == "employee_handbook.pdf"

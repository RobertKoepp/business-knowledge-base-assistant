
import pytest

from answer_generator import build_grounded_response


SAMPLE_RESULTS = [
    {
        "source": "employee_handbook.pdf",
        "page": 3,
        "chunk_id": 1,
        "text": (
            "Employees must submit vacation requests "
            "at least 30 days before planned leave."
        ),
        "relevance_score": 0.82,
    },
    {
        "source": "employee_handbook.pdf",
        "page": 4,
        "chunk_id": 2,
        "text": (
            "Full-time employees receive 15 vacation days "
            "per calendar year."
        ),
        "relevance_score": 0.65,
    },
    {
        "source": "support_policy.docx",
        "page": None,
        "chunk_id": 1,
        "text": (
            "Managers approve requests based on staffing needs."
        ),
        "relevance_score": 0.51,
    },
]


def test_builds_response_from_retrieved_passages():
    result = build_grounded_response(SAMPLE_RESULTS)

    assert result["status"] == "found"
    assert "30 days" in result["answer"]
    assert "15 vacation days" in result["answer"]


def test_preserves_source_references():
    result = build_grounded_response(SAMPLE_RESULTS)

    assert len(result["sources"]) == 3
    assert result["sources"][0]["source"] == (
        "employee_handbook.pdf"
    )
    assert result["sources"][0]["page"] == 3


def test_respects_maximum_source_limit():
    result = build_grounded_response(
        SAMPLE_RESULTS,
        max_sources=1,
    )

    assert len(result["sources"]) == 1


def test_duplicate_sources_are_removed():
    duplicated_results = [
        SAMPLE_RESULTS[0],
        SAMPLE_RESULTS[0],
        SAMPLE_RESULTS[1],
    ]

    result = build_grounded_response(duplicated_results)

    assert len(result["sources"]) == 2


def test_empty_results_return_not_found():
    result = build_grounded_response([])

    assert result["status"] == "not_found"
    assert len(result["sources"]) == 0
    assert "couldn't find" in result["answer"].lower()


def test_invalid_results_are_ignored():
    results = [
        None,
        {"source": "empty.txt", "text": "   "},
        {"source": "invalid.txt", "text": 123},
        SAMPLE_RESULTS[0],
    ]

    result = build_grounded_response(results)

    assert result["status"] == "found"
    assert len(result["sources"]) == 1


def test_invalid_max_sources_is_rejected():
    for invalid_value in [0, -1, 1.5, True]:
        with pytest.raises(
            ValueError,
            match="positive integer",
        ):
            build_grounded_response(
                SAMPLE_RESULTS,
                max_sources=invalid_value,
            )


def test_disclaimer_is_included():
    result = build_grounded_response(SAMPLE_RESULTS)

    assert "disclaimer" in result
    assert "original sources" in result["disclaimer"]

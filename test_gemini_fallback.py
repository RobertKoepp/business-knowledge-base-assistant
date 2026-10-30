
import pytest

from gemini_fallback import generate_with_fallback


SAMPLE_RESULTS = [
    {
        "text": "Full-time employees receive 15 vacation days.",
        "source": "employee_handbook.txt",
        "page": None,
        "chunk_id": 1,
        "relevance_score": 0.82,
    }
]


def fake_extractive_response(results):
    return {
        "status": "found",
        "answer": "Full-time employees receive 15 vacation days.",
        "sources": [
            {
                "source": "employee_handbook.txt",
                "text": results[0]["text"],
            }
        ],
    }


def test_503_error_uses_extractive_fallback():
    def failing_generator(**kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    response = generate_with_fallback(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        generate_answer=failing_generator,
        build_extractive_response=fake_extractive_response,
    )

    assert response["status"] == "fallback"
    assert "15 vacation days" in response["answer"]
    assert "temporarily unavailable" in response["disclaimer"]


def test_429_rate_limit_uses_extractive_fallback():
    def failing_generator(**kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    response = generate_with_fallback(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        generate_answer=failing_generator,
        build_extractive_response=fake_extractive_response,
    )

    assert response["status"] == "fallback"
    assert response["sources"]


def test_successful_gemini_response_does_not_use_fallback():
    expected_response = {
        "status": "found",
        "answer": "Employees receive 15 days. [Source 1]",
        "sources": SAMPLE_RESULTS,
    }

    def successful_generator(**kwargs):
        return expected_response

    def should_not_be_called(results):
        raise AssertionError("Fallback should not be called.")

    response = generate_with_fallback(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        generate_answer=successful_generator,
        build_extractive_response=should_not_be_called,
    )

    assert response is expected_response
    assert response["status"] == "found"


def test_authentication_error_is_not_suppressed():
    def failing_generator(**kwargs):
        raise RuntimeError("401 UNAUTHENTICATED")

    with pytest.raises(RuntimeError, match="401"):
        generate_with_fallback(
            query="How many vacation days?",
            results=SAMPLE_RESULTS,
            generate_answer=failing_generator,
            build_extractive_response=fake_extractive_response,
        )


def test_temporary_error_without_results_is_not_suppressed():
    def failing_generator(**kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError, match="503"):
        generate_with_fallback(
            query="What is the WiFi password?",
            results=[],
            generate_answer=failing_generator,
            build_extractive_response=fake_extractive_response,
        )


def test_temporary_error_without_fallback_evidence_is_not_suppressed():
    def failing_generator(**kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    def empty_extractive_response(results):
        return {
            "status": "not_found",
            "answer": "No relevant information found.",
            "sources": [],
        }

    with pytest.raises(RuntimeError, match="503"):
        generate_with_fallback(
            query="What is the WiFi password?",
            results=SAMPLE_RESULTS,
            generate_answer=failing_generator,
            build_extractive_response=empty_extractive_response,
        )

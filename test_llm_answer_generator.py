
from types import SimpleNamespace

import pytest

from llm_answer_generator import (
    generate_grounded_answer,
    NOT_FOUND_MESSAGE,
    UNVERIFIED_MESSAGE,
)


SAMPLE_RESULTS = [
    {
        "text": (
            "Full-time employees receive 15 vacation days "
            "per calendar year."
        ),
        "source": "employee_handbook.pdf",
        "page": 4,
        "chunk_id": 2,
        "relevance_score": 0.82,
    },
    {
        "text": (
            "Employees must submit vacation requests "
            "at least 30 days before planned leave."
        ),
        "source": "employee_handbook.pdf",
        "page": 5,
        "chunk_id": 3,
        "relevance_score": 0.78,
    },
]


class FakeModels:
    """Simulate Gemini responses without making API calls."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.last_model = None
        self.last_contents = None

    def generate_content(self, model, contents):
        self.last_model = model
        self.last_contents = contents

        return SimpleNamespace(text=self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


def test_generates_answer_with_valid_citation():
    client = FakeClient(
        "Full-time employees receive 15 vacation days "
        "per year. [Source 1]"
    )

    response = generate_grounded_answer(
        query="How many vacation days do employees receive?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    assert response["status"] == "found"
    assert "[Source 1]" in response["answer"]
    assert len(response["sources"]) == 2
    assert response["sources"][0]["source"] == (
        "employee_handbook.pdf"
    )
    assert response["model"] == "gemini-3.8-flash"


def test_sends_question_and_document_context_to_gemini():
    client = FakeClient(
        "Employees receive 15 vacation days. [Source 1]"
    )

    generate_grounded_answer(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    prompt = client.models.last_contents

    assert "How many vacation days?" in prompt
    assert "15 vacation days" in prompt
    assert "employee_handbook.pdf" in prompt
    assert "Use ONLY" in prompt or "using ONLY" in prompt


def test_uses_default_model():
    client = FakeClient("Employees receive 15 days. [Source 1]")

    generate_grounded_answer(
        query="How many days?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    assert client.models.last_model == "gemini-3.8-flash"


def test_returns_not_found_when_no_results():
    response = generate_grounded_answer(
        query="What is the WiFi password?",
        results=[],
    )

    assert response["status"] == "not_found"
    assert response["answer"] == NOT_FOUND_MESSAGE
    assert response["sources"] == []


def test_returns_not_found_when_sources_are_invalid():
    results = [
        {
            "text": "Some document text.",
        }
    ]

    response = generate_grounded_answer(
        query="What does the document say?",
        results=results,
    )

    assert response["status"] == "not_found"
    assert response["sources"] == []


def test_returns_not_found_when_gemini_has_insufficient_evidence():
    client = FakeClient("NOT_ENOUGH_EVIDENCE")

    response = generate_grounded_answer(
        query="What is the WiFi password?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    assert response["status"] == "not_found"
    assert response["answer"] == NOT_FOUND_MESSAGE
    assert response["sources"] == []


def test_rejects_answer_without_citations():
    client = FakeClient(
        "Full-time employees receive 15 vacation days."
    )

    response = generate_grounded_answer(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    assert response["status"] == "unverified"
    assert response["answer"] == UNVERIFIED_MESSAGE


def test_rejects_invalid_citation_numbers():
    client = FakeClient(
        "Employees receive 15 vacation days. [Source 99]"
    )

    response = generate_grounded_answer(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        client=client,
    )

    assert response["status"] == "unverified"


def test_rejects_empty_gemini_response():
    client = FakeClient("   ")

    with pytest.raises(RuntimeError):
        generate_grounded_answer(
            query="How many vacation days?",
            results=SAMPLE_RESULTS,
            client=client,
        )


def test_respects_maximum_source_limit():
    client = FakeClient(
        "Employees receive 15 vacation days. [Source 1]"
    )

    response = generate_grounded_answer(
        query="How many vacation days?",
        results=SAMPLE_RESULTS,
        max_sources=1,
        client=client,
    )

    assert len(response["sources"]) == 1


def test_duplicate_sources_are_removed():
    duplicated_results = [
        SAMPLE_RESULTS[0],
        SAMPLE_RESULTS[0],
    ]

    client = FakeClient(
        "Employees receive 15 vacation days. [Source 1]"
    )

    response = generate_grounded_answer(
        query="How many vacation days?",
        results=duplicated_results,
        client=client,
    )

    assert len(response["sources"]) == 1


def test_rejects_empty_query():
    with pytest.raises(ValueError):
        generate_grounded_answer(
            query=" ",
            results=SAMPLE_RESULTS,
            client=FakeClient("Test [Source 1]"),
        )


def test_rejects_invalid_results_type():
    with pytest.raises(ValueError):
        generate_grounded_answer(
            query="Test question",
            results="not a list",
            client=FakeClient("Test [Source 1]"),
        )


def test_rejects_invalid_max_sources():
    with pytest.raises(ValueError):
        generate_grounded_answer(
            query="Test question",
            results=SAMPLE_RESULTS,
            max_sources=0,
            client=FakeClient("Test [Source 1]"),
        )


def test_requires_api_key_when_client_is_not_supplied(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate_grounded_answer(
            query="How many vacation days?",
            results=SAMPLE_RESULTS,
        )

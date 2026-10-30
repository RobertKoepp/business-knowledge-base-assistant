
from io import BytesIO

from streamlit.testing.v1 import AppTest

from document_processor import process_document
from search_engine import search_chunks
from answer_generator import build_grounded_response


SAMPLE_HANDBOOK = """
EMPLOYEE HANDBOOK - VACATION POLICY

Vacation Requests:
Employees must submit vacation requests at least
30 days before their planned leave.
Requests should be submitted to their direct manager
for approval.

Vacation Balance:
Full-time employees receive 15 vacation days
per calendar year.

Approval Process:
Managers review vacation requests based on staffing
requirements and business needs.
"""


def make_sample_document():
    """Create a simulated uploaded employee handbook."""
    uploaded = BytesIO(
        SAMPLE_HANDBOOK.encode("utf-8")
    )
    uploaded.name = "sample_employee_handbook.txt"

    return uploaded


def test_app_opens_without_errors():
    app = AppTest.from_file("app.py").run()

    assert not app.exception


def test_app_displays_document_upload_section():
    app = AppTest.from_file("app.py").run()

    assert any(
        "Upload business documents" in heading.value
        for heading in app.header
    )


def test_processing_without_upload_shows_warning():
    app = AppTest.from_file("app.py").run()

    app.button[0].click().run()

    assert len(app.warning) > 0
    assert "upload" in app.warning[0].value.lower()


def test_document_processing_and_search_work_together():
    uploaded = make_sample_document()

    chunks = process_document(uploaded)

    results = search_chunks(
        chunks,
        "How much notice is required for vacation requests?",
        top_k=3,
    )

    assert len(chunks) > 0
    assert len(results) > 0

    assert results[0]["source"] == (
        "sample_employee_handbook.txt"
    )

    assert "30 days" in results[0]["text"].lower()


def test_search_results_preserve_source_information():
    uploaded = make_sample_document()

    chunks = process_document(uploaded)

    results = search_chunks(
        chunks,
        "vacation requests",
    )

    assert len(results) > 0

    result = results[0]

    assert result["source"] == "sample_employee_handbook.txt"
    assert result["page"] is None
    assert result["chunk_id"] == 1
    assert result["relevance_score"] > 0


def test_irrelevant_question_returns_no_results():
    uploaded = make_sample_document()

    chunks = process_document(uploaded)

    results = search_chunks(
        chunks,
        "quantum spacecraft telescope",
    )

    assert results == []


def test_document_to_grounded_response_pipeline():
    uploaded = make_sample_document()

    # Extract and organize the document.
    chunks = process_document(uploaded)

    # Retrieve relevant passages.
    results = search_chunks(
        chunks,
        "How much notice is required for vacation requests?",
        top_k=3,
    )

    # Assemble a source-backed response.
    response = build_grounded_response(results)

    assert response["status"] == "found"
    assert "30 days" in response["answer"]
    assert len(response["sources"]) > 0
    assert (
        response["sources"][0]["source"]
        == "sample_employee_handbook.txt"
    )


def test_unanswered_question_returns_not_found():
    uploaded = make_sample_document()

    chunks = process_document(uploaded)

    results = search_chunks(
        chunks,
        "quantum spacecraft telescope",
    )

    response = build_grounded_response(results)

    assert response["status"] == "not_found"
    assert len(response["sources"]) == 0


def test_grounded_response_preserves_document_references():
    uploaded = make_sample_document()

    chunks = process_document(uploaded)

    results = search_chunks(
        chunks,
        "vacation requests",
    )

    response = build_grounded_response(results)

    assert response["status"] == "found"

    source = response["sources"][0]

    assert source["source"] == "sample_employee_handbook.txt"
    assert source["page"] is None
    assert source["chunk_id"] == 1
    assert source["relevance_score"] > 0

    assert "original sources" in response["disclaimer"]

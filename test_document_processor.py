
from io import BytesIO

import pytest
from docx import Document

import document_processor as dp


def make_upload(content, filename):
    """Create a file-like object that simulates an uploaded file."""
    uploaded = BytesIO(content)
    uploaded.name = filename
    return uploaded


def create_docx():
    """Create a sample Word document for testing."""
    document = Document()

    document.add_paragraph(
        "Employees should submit vacation requests in advance."
    )

    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Policy"
    table.cell(0, 1).text = "30 days notice"

    buffer = BytesIO()
    document.save(buffer)

    return buffer.getvalue()


def test_extract_txt_document():
    content = b"Employees should submit vacation requests early."
    uploaded = make_upload(content, "employee_policy.txt")

    result = dp.extract_document(uploaded)

    assert result["filename"] == "employee_policy.txt"
    assert len(result["pages"]) == 1
    assert "vacation requests" in result["pages"][0]["text"]


def test_extract_docx_document():
    uploaded = make_upload(
        create_docx(),
        "employee_policy.docx",
    )

    result = dp.extract_document(uploaded)

    extracted_text = result["pages"][0]["text"]

    assert result["filename"] == "employee_policy.docx"
    assert "vacation requests" in extracted_text
    assert "30 days notice" in extracted_text


def test_extract_pdf_document(monkeypatch):
    """Test PDF extraction without needing to generate a real PDF."""

    class FakePage:
        def extract_text(self):
            return "The company provides 15 vacation days."

    class FakeReader:
        def __init__(self, file):
            self.pages = [FakePage()]

    monkeypatch.setattr(dp, "PdfReader", FakeReader)

    uploaded = make_upload(b"sample PDF", "policy.pdf")

    result = dp.extract_document(uploaded)

    assert result["filename"] == "policy.pdf"
    assert result["pages"][0]["page"] == 1
    assert "15 vacation days" in result["pages"][0]["text"]


def test_reject_unsupported_file():
    uploaded = make_upload(
        b"sample image",
        "policy.png",
    )

    with pytest.raises(ValueError, match="Unsupported file type"):
        dp.extract_document(uploaded)


def test_reject_empty_text_file():
    uploaded = make_upload(b"   ", "empty.txt")

    with pytest.raises(ValueError, match="No readable text"):
        dp.extract_document(uploaded)


def test_reject_pdf_without_extractable_text(monkeypatch):
    """Verify that image-only or empty PDFs are rejected clearly."""

    class FakePage:
        def extract_text(self):
            return ""

    class FakeReader:
        def __init__(self, file):
            self.pages = [FakePage()]

    monkeypatch.setattr(dp, "PdfReader", FakeReader)

    uploaded = make_upload(b"sample PDF", "scanned.pdf")

    with pytest.raises(ValueError, match="No readable text"):
        dp.extract_document(uploaded)


def test_chunk_text_splits_content():
    text = "one two three four five six seven eight"

    chunks = dp.chunk_text(
        text,
        chunk_size=5,
        overlap=2,
    )

    assert len(chunks) == 2
    assert chunks[0] == "one two three four five"
    assert chunks[1] == "four five six seven eight"


def test_chunk_text_handles_empty_content():
    chunks = dp.chunk_text("   ")

    assert chunks == []


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="Chunk size"):
        dp.chunk_text("some text", chunk_size=0)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="Overlap"):
        dp.chunk_text(
            "some example text",
            chunk_size=5,
            overlap=5,
        )


def test_process_document_preserves_source_information():
    text = (
        "one two three four five six seven eight "
        "nine ten eleven twelve"
    )

    uploaded = make_upload(
        text.encode("utf-8"),
        "company_handbook.txt",
    )

    chunks = dp.process_document(
        uploaded,
        chunk_size=5,
        overlap=2,
    )

    assert len(chunks) == 4

    for index, chunk in enumerate(chunks, start=1):
        assert chunk["source"] == "company_handbook.txt"
        assert chunk["page"] is None
        assert chunk["chunk_id"] == index
        assert chunk["text"]

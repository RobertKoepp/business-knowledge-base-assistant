
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_pdf(file):
    """Extract text from each page of a PDF."""
    try:
        file.seek(0)
        reader = PdfReader(file)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                })

        return pages

    except Exception as error:
        raise ValueError(
            f"Unable to read PDF: {error}"
        ) from error


def extract_docx(file):
    """Extract paragraphs and table contents from a Word document."""
    try:
        file.seek(0)
        document = Document(file)

        text_parts = []

        # Extract regular paragraphs.
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

        # Extract table contents.
        for table in document.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cells:
                    text_parts.append(" | ".join(cells))

        full_text = "\n".join(text_parts)

        if not full_text.strip():
            raise ValueError(
                "No readable text was found in this Word document."
            )

        return [{
            "page": None,
            "text": full_text,
        }]

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to read Word document: {error}"
        ) from error


def extract_txt(file):
    """Extract text from a plain text file."""
    try:
        file.seek(0)
        raw_text = file.read()

        text = raw_text.decode(
            "utf-8",
            errors="replace",
        )

        if not text.strip():
            raise ValueError(
                "No readable text was found in this text file."
            )

        return [{
            "page": None,
            "text": text,
        }]

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to read text file: {error}"
        ) from error


def extract_document(file):
    """Identify the file type and extract its contents."""
    filename = Path(file.name).name
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        pages = extract_pdf(file)

    elif extension == ".docx":
        pages = extract_docx(file)

    elif extension == ".txt":
        pages = extract_txt(file)

    else:
        raise ValueError(
            "Unsupported file type. Please upload PDF, DOCX, or TXT."
        )

    if not pages:
        raise ValueError(
            "No readable text was found in this document. "
            "Scanned PDFs without selectable text are not supported."
        )

    return {
        "filename": filename,
        "pages": pages,
    }


def chunk_text(text, chunk_size=150, overlap=25):
    """
    Split text into smaller sections using word counts.

    chunk_size: Maximum words in each section.
    overlap: Number of repeated words between sections.
    """
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be zero or greater and smaller than chunk size."
        )

    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(words):
        section = words[start:start + chunk_size]

        if not section:
            break

        chunks.append(" ".join(section))

        # Stop once the current chunk reaches the end of the text.
        if start + chunk_size >= len(words):
            break

        start += step

    return chunks


def process_document(file, chunk_size=150, overlap=25):
    """Extract a document and return searchable text chunks."""
    document = extract_document(file)

    filename = document["filename"]
    all_chunks = []

    for page_data in document["pages"]:
        text = page_data["text"]
        page_number = page_data["page"]

        sections = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for section in sections:
            all_chunks.append({
                "source": filename,
                "page": page_number,
                "chunk_id": len(all_chunks) + 1,
                "text": section,
            })

    return all_chunks

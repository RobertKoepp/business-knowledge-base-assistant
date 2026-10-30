# Business Knowledge Base Assistant

An AI-powered document search and question-answering application that helps businesses find information across internal documents.

Built with Python, Streamlit, Sentence Transformers, and the Google Gemini API, this application combines semantic search with Retrieval-Augmented Generation (RAG) to retrieve relevant document passages and generate contextual answers with source references.

## Overview

Businesses often store important information across employee handbooks, IT policies, expense procedures, and other documents. Finding specific information across these files can be time-consuming.

The Business Knowledge Base Assistant provides a centralized way to upload documents, search their contents, and ask questions using natural language.

Users can choose between keyword search, semantic search, extractive responses, and AI-generated answers using Google Gemini.

## Features

### Document Processing
- Upload PDF, DOCX, and TXT files.
- Extract text and organize it into searchable chunks.
- Preserve document names, page numbers, and section identifiers.

### Keyword Search
- Uses TF-IDF to rank passages based on matching words and phrases.
- Provides a traditional search option without requiring an LLM.

### Semantic Search
- Uses Sentence Transformers and the all-MiniLM-L6-v2 embedding model.
- Converts document passages and user questions into vector embeddings.
- Uses cosine similarity to retrieve passages based on meaning.
- Supports configurable similarity thresholds.

### Gemini AI Answer Generation
- Integrates with the Google Gemini API.
- Generates natural-language answers from retrieved document passages.
- Includes source-number references in generated answers.
- Returns an insufficient-evidence response when the retrieved content cannot answer the question.
- Validates that generated answers include references to available source numbers.

### Reliability and Error Handling
- Falls back to retrieved document passages when Gemini experiences recognized temporary service errors or rate limits.
- Handles unsupported questions.
- Preserves extractive responses without requiring an API call.
- Includes automated tests for search, answer generation, and fallback behavior.

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow.

```mermaid
flowchart TD
    A[Document Upload] --> B[Document Processing]
    B --> C[Text Extraction]
    C --> D[Document Chunking]
    D --> E[Search Engine]
    E --> F[Keyword Search - TF-IDF]
    E --> G[Semantic Search - Embeddings]
    F --> H[Retrieved Passages]
    G --> H
    H --> I[Google Gemini API]
    I --> J[Grounded AI Answer]
    J --> K[Source References]
```

### Retrieval and Generation Process

1. Users upload business documents.
2. The document processor extracts and organizes the content.
3. The search engine retrieves passages relevant to the user's question.
4. Retrieved passages and source information are provided to Gemini.
5. Gemini generates an answer using the supplied document context.
6. The application checks that the answer references available source numbers and displays supporting passages.

The application also supports extractive responses, which display retrieved document passages without using an LLM.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Application development |
| Streamlit | Interactive web interface |
| Google Gemini API | AI answer generation |
| Google Gen AI SDK | Gemini API integration |
| Sentence Transformers | Semantic embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| NumPy | Vector operations |
| scikit-learn | TF-IDF keyword search |
| PyPDF | PDF text extraction |
| python-docx | Word document processing |
| pytest | Automated testing |

## Project Structure

```text
business-knowledge-base-assistant/
|
|-- app.py
|-- document_processor.py
|-- search_engine.py
|-- semantic_search.py
|-- answer_generator.py
|-- llm_answer_generator.py
|-- gemini_fallback.py
|
|-- test_app.py
|-- test_document_processor.py
|-- test_search.py
|-- test_search_modes.py
|-- test_semantic_search.py
|-- test_answer_generator.py
|-- test_llm_answer_generator.py
|-- test_gemini_fallback.py
|
|-- sample_employee_handbook.txt
|-- sample_it_and_expense_policy.txt
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## Installation

### Requirements

- Python 3.12
- pip
- A Google Gemini API key for AI-generated answers

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/business-knowledge-base-assistant.git

cd business-knowledge-base-assistant
```

Replace YOUR-GITHUB-USERNAME with your GitHub username.

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Obtain a Gemini API key through Google AI Studio:

https://aistudio.google.com/apikey

Set your API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Never commit API keys or other credentials to GitHub.

The Gemini API key is required for AI-generated answers. Keyword search, semantic search, and extractive responses do not require Gemini API access.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where the application can be accessed.

The first semantic search may take longer because the embedding model needs to be downloaded and loaded.

## Running Tests

The project includes automated tests covering document processing, keyword search, semantic retrieval, answer generation, application behavior, and Gemini fallback handling.

Run the complete test suite:

```bash
python -m pytest -q
```

Latest locally verified test result:

```text
89 passed
```

Run individual test modules:

```bash
python -m pytest -q test_document_processor.py
python -m pytest -q test_search.py
python -m pytest -q test_semantic_search.py
python -m pytest -q test_answer_generator.py
python -m pytest -q test_llm_answer_generator.py
python -m pytest -q test_gemini_fallback.py
python -m pytest -q test_app.py
python -m pytest -q test_search_modes.py
```

## Example Use Cases

The project includes fictional sample business documents covering employee policies, IT support, and expense reimbursement.

Example questions:

- How many vacation days do full-time employees receive?
- How far in advance must vacation be requested?
- Who should employees contact for technical support?
- What are the requirements for expense reimbursement?
- What is the company WiFi password?

Questions that cannot be answered using retrieved document passages should receive an insufficient-evidence response.

## Security and Limitations

- Gemini API credentials should be stored in environment variables or an approved secrets manager.
- Documents submitted for AI-generated answers are transmitted to the configured Gemini API. Do not upload confidential business information unless your data-handling arrangements authorize it.
- This application is a portfolio prototype and has not been validated for production use.
- Generated answers may contain errors or misinterpret document content.
- Citation-number validation confirms that referenced source identifiers exist. It does not independently prove every generated statement is supported by its cited passage.
- Semantic similarity scores represent retrieval relevance, not factual confidence.
- PDF extraction quality depends on document structure. Scanned documents may require OCR support.
- The application does not currently implement user authentication, role-based access, persistent document storage, or a production vector database.

## Future Improvements

- Chat-style interface and conversation history.
- Improved citation verification.
- Persistent vector storage.
- Multi-user access and authentication.
- Document management and deletion.
- Production deployment.
- Expanded evaluation using realistic business document collections.

## Author

Developed as an AI application and RAG portfolio project.

Demonstrates practical experience with Python application development, document processing, semantic retrieval, LLM integration, prompt design, automated testing, and AI reliability engineering.

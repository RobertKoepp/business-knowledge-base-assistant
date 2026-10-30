
from functools import lru_cache

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load and cache the embedding model.

    The model is loaded only once per Python process.
    """
    try:
        from sentence_transformers import SentenceTransformer

    except ImportError as error:
        raise ImportError(
            "Sentence Transformers is not installed. "
            "Activate the .venv-semantic environment and install "
            "the required dependencies."
        ) from error

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def semantic_search(
    chunks,
    query,
    top_k=5,
    min_score=0.25,
    model=None,
):
    """
    Search document chunks using semantic embeddings.

    Args:
        chunks: Document chunks containing source information.
        query: The user's search question.
        top_k: Maximum number of results.
        min_score: Minimum cosine similarity score.
        model: Optional embedding model for testing.

    Returns:
        Ranked document chunks with semantic relevance scores.
    """

    # Validate query.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a search question.")

    # Validate number of results.
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    # Validate similarity threshold.
    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not np.isfinite(min_score)
        or not -1 <= min_score <= 1
    ):
        raise ValueError(
            "min_score must be a finite number between -1 and 1."
        )

    # Remove invalid and empty chunks.
    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not valid_chunks:
        return []

    # Load the cached model unless one was supplied.
    if model is None:
        model = load_embedding_model()

    documents = [
        chunk["text"]
        for chunk in valid_chunks
    ]

    # Convert document passages into normalized embeddings.
    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Convert the user's question into an embedding.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Calculate cosine similarity.
    # The embeddings are normalized, so dot product is equivalent.
    similarities = np.asarray(
        document_embeddings
    ) @ np.asarray(query_embedding)

    # Rank results by similarity.
    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        # Exclude passages below the relevance threshold.
        if score < min_score:
            continue

        result = dict(valid_chunks[index])

        result["relevance_score"] = round(score, 4)
        result["semantic_score"] = round(score, 4)

        results.append(result)

        if len(results) >= top_k:
            break

    return results

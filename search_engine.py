
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_chunks(chunks, query, top_k=5):
    """
    Search document chunks using TF-IDF and cosine similarity.

    Args:
        chunks: List of document chunks containing source information.
        query: The user's search question.
        top_k: Maximum number of results to return.

    Returns:
        Ranked list of matching document chunks with relevance scores.
    """

    # Validate the search query.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a search question.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Remove empty or invalid document chunks.
    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    # Return an empty list if there is nothing to search.
    if not valid_chunks:
        return []

    documents = [
        chunk["text"]
        for chunk in valid_chunks
    ]

    # Convert document text into numerical TF-IDF vectors.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        document_vectors = vectorizer.fit_transform(documents)

    except ValueError:
        # Handles documents without usable searchable words.
        return []

    # Convert the user's query using the same vocabulary.
    query_vector = vectorizer.transform([query])

    # Return no results if the query contains no searchable words.
    if query_vector.nnz == 0:
        return []

    # Calculate similarity between the query and every document chunk.
    similarities = cosine_similarity(
        query_vector,
        document_vectors,
    ).flatten()

    # Rank results from most relevant to least relevant.
    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        # Ignore chunks with no matching terms.
        if score <= 0:
            continue

        result = dict(valid_chunks[index])

        # Preserve document information and add a relevance score.
        result["relevance_score"] = round(score, 4)

        results.append(result)

        if len(results) >= top_k:
            break

    return results

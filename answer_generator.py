
def build_grounded_response(results, max_sources=3):
    """
    Build an evidence-based response from retrieved document passages.

    This is extractive response assembly, not AI-generated summarization.
    It only displays information found in the retrieved passages.
    """

    if not isinstance(results, list):
        raise ValueError("Results must be provided as a list.")

    if (
        not isinstance(max_sources, int)
        or isinstance(max_sources, bool)
        or max_sources <= 0
    ):
        raise ValueError("Maximum sources must be a positive integer.")

    if not results:
        return {
            "status": "not_found",
            "answer": (
                "I couldn't find relevant information in the "
                "uploaded documents. Try different search terms "
                "or upload another document."
            ),
            "sources": [],
        }

    valid_sources = []
    seen_sources = set()

    for result in results:
        if not isinstance(result, dict):
            continue

        text = result.get("text")
        source = result.get("source")
        page = result.get("page")
        chunk_id = result.get("chunk_id")

        if not isinstance(text, str) or not text.strip():
            continue

        if not isinstance(source, str) or not source.strip():
            continue

        source_key = (source, page, chunk_id)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        valid_sources.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "text": text.strip(),
            "relevance_score": result.get("relevance_score"),
        })

        if len(valid_sources) >= max_sources:
            break

    if not valid_sources:
        return {
            "status": "not_found",
            "answer": (
                "I couldn't find usable document passages "
                "to answer your question."
            ),
            "sources": [],
        }

    # Assemble a response using only retrieved document text.
    answer_parts = [
        "The uploaded documents contain the following "
        "relevant information:"
    ]

    for index, source in enumerate(valid_sources, start=1):
        answer_parts.append(
            f"{index}. {source['text']}"
        )

    answer = "\n\n".join(answer_parts)

    return {
        "status": "found",
        "answer": answer,
        "sources": valid_sources,
        "disclaimer": (
            "This response displays retrieved document passages "
            "rather than an independently generated conclusion. "
            "Review the original sources before making decisions."
        ),
    }

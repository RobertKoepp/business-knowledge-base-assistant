
def is_temporary_gemini_error(error):
    """Identify temporary Gemini service and rate-limit errors."""

    status_code = getattr(error, "status_code", None)

    if status_code in (429, 500, 502, 503, 504):
        return True

    error_text = str(error).upper()

    temporary_error_markers = (
        "503 UNAVAILABLE",
        "429 RESOURCE_EXHAUSTED",
        "500 INTERNAL",
        "502 BAD GATEWAY",
        "504 GATEWAY TIMEOUT",
        "SERVICE UNAVAILABLE",
        "TOO MANY REQUESTS",
        "RESOURCE_EXHAUSTED",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "GATEWAY TIMEOUT",
    )

    return any(
        marker in error_text
        for marker in temporary_error_markers
    )


def generate_with_fallback(
    query,
    results,
    generate_answer,
    build_extractive_response,
    max_sources=5,
):
    """
    Generate a Gemini answer, falling back to retrieved passages
    when Gemini experiences a temporary service failure.

    Authentication and other non-temporary errors are not suppressed.
    """

    try:
        return generate_answer(
            query=query,
            results=results,
            max_sources=max_sources,
        )

    except Exception as error:
        if not is_temporary_gemini_error(error) or not results:
            raise

        fallback_response = build_extractive_response(results)

        if fallback_response.get("status") != "found":
            raise

        fallback_response = dict(fallback_response)

        fallback_response["status"] = "fallback"

        fallback_response["disclaimer"] = (
            "Gemini was temporarily unavailable or rate-limited. "
            "This response displays retrieved document passages "
            "instead of an AI-generated conclusion."
        )

        return fallback_response


import os

import streamlit as st

from document_processor import process_document
from search_engine import search_chunks
from answer_generator import build_grounded_response
from llm_answer_generator import generate_grounded_answer
from gemini_fallback import generate_with_fallback


st.set_page_config(
    page_title="Business Knowledge Base Assistant",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# APPLICATION STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-weight: 750;
            letter-spacing: -0.03em;
        }

        div.stButton > button {
            border-radius: 10px;
            min-height: 2.7rem;
            font-weight: 600;
        }

        [data-testid="stTextArea"] textarea {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "chunks": [],
    "processed_files": [],
    "processing_errors": [],
    "search_results": [],
    "answer_response": None,
    "search_error": "",
    "search_performed": False,
    "search_method_used": "",
    "response_mode_used": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("📚 Business Knowledge Base Assistant")

st.caption(
    "Search business documents and generate "
    "AI-powered answers grounded in your information."
)

st.info(
    "Upload business documents, retrieve relevant passages, "
    "and optionally use Gemini to generate natural-language "
    "answers supported by your documents."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About this tool")

    st.write(
        "Upload business documents and search their contents "
        "to find relevant policies, procedures, and information."
    )

    st.divider()

    st.subheader("Supported formats")

    st.write("- PDF")
    st.write("- Word documents (.docx)")
    st.write("- Plain text (.txt)")

    st.divider()

    st.subheader("Search methods")

    st.markdown(
        """
        **Keyword search**

        Uses TF-IDF to rank matching text.

        **Semantic search**

        Uses a locally running embedding model to compare meaning.
        """
    )

    st.divider()

    st.subheader("Answer generation")

    st.markdown(
        """
        **Extractive response**

        Displays retrieved document passages.

        **Gemini AI response**

        Generates a natural-language answer using retrieved passages.
        """
    )

    st.divider()

    st.subheader("Important limitations")

    st.caption(
        "AI answers can contain errors or misunderstand documents. "
        "Similarity scores are not factual confidence scores. "
        "Always verify important information against original sources."
    )


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.header("1. Upload business documents")

uploaded_files = st.file_uploader(
    "Choose your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Upload PDF, DOCX, or TXT files.",
)


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

if st.button("Process documents", type="primary"):

    st.session_state.chunks = []
    st.session_state.processed_files = []
    st.session_state.processing_errors = []
    st.session_state.search_results = []
    st.session_state.answer_response = None
    st.session_state.search_performed = False
    st.session_state.search_error = ""

    if not uploaded_files:

        st.warning("Please upload at least one document.")

    else:

        all_chunks = []
        successful_files = []
        errors = []

        with st.spinner("Extracting and organizing documents..."):

            for uploaded_file in uploaded_files:

                try:

                    document_chunks = process_document(uploaded_file)

                    if not document_chunks:

                        errors.append(
                            f"{uploaded_file.name}: "
                            "No searchable text found."
                        )
                        continue

                    all_chunks.extend(document_chunks)
                    successful_files.append(uploaded_file.name)

                except ValueError as error:

                    errors.append(
                        f"{uploaded_file.name}: {error}"
                    )

                except Exception:

                    errors.append(
                        f"{uploaded_file.name}: "
                        "An unexpected processing error occurred."
                    )

        st.session_state.chunks = all_chunks
        st.session_state.processed_files = successful_files
        st.session_state.processing_errors = errors


# --------------------------------------------------
# DISPLAY DOCUMENT PROCESSING RESULTS
# --------------------------------------------------

if st.session_state.processed_files:

    st.success(
        f"Successfully processed "
        f"{len(st.session_state.processed_files)} document(s) "
        f"into {len(st.session_state.chunks)} searchable section(s)."
    )

    with st.expander("View processed documents"):

        for filename in st.session_state.processed_files:
            st.write(f"- {filename}")


if st.session_state.processing_errors:

    for error in st.session_state.processing_errors:
        st.warning(error)


# --------------------------------------------------
# KNOWLEDGE BASE SEARCH
# --------------------------------------------------

if st.session_state.chunks:

    st.divider()

    st.header("2. Search your knowledge base")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Uploaded documents",
            len(st.session_state.processed_files),
        )

    with col2:

        st.metric(
            "Searchable sections",
            len(st.session_state.chunks),
        )


    # ----------------------------------------------
    # SEARCH METHOD SELECTION
    # ----------------------------------------------

    search_method = st.radio(
        "Choose a search method",
        [
            "Keyword search (TF-IDF)",
            "Semantic search (Embeddings)",
        ],
        horizontal=True,
        key="search_method",
    )

    if search_method == "Semantic search (Embeddings)":

        st.caption(
            "Semantic search uses the locally installed "
            "all-MiniLM-L6-v2 embedding model. "
            "The model runs on your computer's CPU."
        )

        min_score = st.slider(
            "Minimum semantic similarity",
            min_value=0.0,
            max_value=0.8,
            value=0.25,
            step=0.05,
            help=(
                "Passages below this similarity score will be excluded. "
                "This is a retrieval threshold, not factual confidence."
            ),
        )

    else:

        min_score = 0.25

        st.caption(
            "Keyword search uses TF-IDF to rank passages "
            "based on matching words and phrases."
        )


    # ----------------------------------------------
    # RESPONSE MODE SELECTION
    # ----------------------------------------------

    st.divider()

    st.subheader("How should your answer be generated?")

    response_mode = st.radio(
        "Choose a response type",
        [
            "Extractive response (retrieved passages)",
            "Gemini AI answer",
        ],
        horizontal=True,
        key="response_mode",
    )

    if response_mode == "Gemini AI answer":

        if os.getenv("GEMINI_API_KEY"):

            st.success("Gemini API key detected.")

        else:

            st.warning(
                "Gemini API key not detected. "
                "Load GEMINI_API_KEY in your terminal before "
                "launching the application."
            )

        st.caption(
            "Gemini uses retrieved passages to generate an answer. "
            "AI-generated responses may contain errors."
        )

    else:

        st.caption(
            "Extractive mode displays the retrieved document passages "
            "without generating an independent AI conclusion."
        )


    # ----------------------------------------------
    # USER QUESTION
    # ----------------------------------------------

    query = st.text_input(
        "What would you like to know?",
        placeholder=(
            "Example: How much notice is required for vacation requests?"
        ),
        key="search_query",
    )

    top_k = st.selectbox(
        "Number of results",
        [3, 5, 10],
        index=1,
        key="results_limit",
    )


    # ----------------------------------------------
    # SEARCH BUTTON
    # ----------------------------------------------

    if st.button("Search documents", type="primary"):

        st.session_state.search_error = ""
        st.session_state.search_results = []
        st.session_state.answer_response = None
        st.session_state.search_performed = True

        st.session_state.search_method_used = search_method
        st.session_state.response_mode_used = response_mode

        try:

            with st.spinner("Searching your documents..."):

                # ----------------------------------
                # RETRIEVE RELEVANT DOCUMENT PASSAGES
                # ----------------------------------

                if search_method == "Semantic search (Embeddings)":

                    from semantic_search import semantic_search

                    results = semantic_search(
                        st.session_state.chunks,
                        query,
                        top_k=top_k,
                        min_score=min_score,
                    )

                else:

                    results = search_chunks(
                        st.session_state.chunks,
                        query,
                        top_k=top_k,
                    )

                st.session_state.search_results = results


                # ----------------------------------
                # GENERATE RESPONSE
                # ----------------------------------

                if response_mode == "Gemini AI answer":

                    if not os.getenv("GEMINI_API_KEY"):

                        raise RuntimeError(
                            "GEMINI_API_KEY is not set. "
                            "Load your Gemini API key in the terminal "
                            "and restart the application."
                        )

                    with st.spinner(
                        "Gemini is generating your answer..."
                    ):

                        response = generate_with_fallback(
                            query=query,
                            results=results,
                            generate_answer=generate_grounded_answer,
                            build_extractive_response=build_grounded_response,
                            max_sources=5,
                        )

                else:

                    response = build_grounded_response(results)


                st.session_state.answer_response = response


        # ------------------------------------------
        # ERROR HANDLING
        # ------------------------------------------

        except ValueError as error:

            st.session_state.search_error = str(error)

        except Exception as error:

            st.session_state.search_error = (
                f"Search or answer generation could not be completed: "
                f"{error}"
            )


    # --------------------------------------------------
    # DISPLAY SEARCH AND ANSWER RESULTS
    # --------------------------------------------------

    if st.session_state.search_error:

        st.error(st.session_state.search_error)

    elif st.session_state.search_performed:

        response = st.session_state.answer_response

        st.divider()

        st.header("3. Knowledge base response")


        # ----------------------------------------------
        # SUCCESSFUL AI OR EXTRACTIVE RESPONSE
        # ----------------------------------------------

        if response and response["status"] in (
            "found",
            "fallback",
        ):

            if response["status"] == "fallback":

                st.warning(
                    "Gemini is temporarily unavailable or rate-limited. "
                    "Showing retrieved document passages instead."
                )

            elif (
                st.session_state.response_mode_used
                == "Gemini AI answer"
            ):

                st.success(
                    f"AI-generated answer using "
                    f"{response.get('model', 'Gemini')}."
                )

            else:

                st.success(
                    f"Relevant information retrieved using "
                    f"{st.session_state.search_method_used}."
                )


            st.markdown(response["answer"])

            if response.get("disclaimer"):

                st.caption(response["disclaimer"])


        # ----------------------------------------------
        # UNVERIFIED AI RESPONSE
        # ----------------------------------------------

        elif response and response["status"] == "unverified":

            st.warning(response["answer"])


        # ----------------------------------------------
        # NO ANSWER FOUND
        # ----------------------------------------------

        else:

            st.warning(
                response["answer"]
                if response
                else "No response could be generated."
            )


        # ----------------------------------------------
        # SUPPORTING SOURCES
        # ----------------------------------------------

        if response and response.get("sources"):

            st.subheader("Supporting sources")

            for index, source in enumerate(
                response["sources"],
                start=1,
            ):

                with st.expander(
                    f"Source {index}: {source['source']}"
                ):

                    st.write(
                        "**Document:**",
                        source["source"],
                    )

                    if source.get("page") is not None:

                        st.write(
                            "**Page:**",
                            source["page"],
                        )

                    else:

                        st.write(
                            "**Page:** Not available"
                        )

                    st.write(
                        "**Section:**",
                        source.get("chunk_id", "Not available"),
                    )

                    if source.get("relevance_score") is not None:

                        st.write(
                            "**Relevance score:**",
                            source["relevance_score"],
                        )

                    st.write("**Retrieved passage:**")

                    st.write(source["text"])


        # ----------------------------------------------
        # ORIGINAL RETRIEVED PASSAGES
        # ----------------------------------------------

        if st.session_state.search_results:

            st.divider()

            st.subheader("4. Retrieved document passages")

            for index, result in enumerate(
                st.session_state.search_results,
                start=1,
            ):

                score = result.get("relevance_score")

                score_label = (
                    f"{score:.3f} similarity"
                    if isinstance(score, (int, float))
                    else "score unavailable"
                )

                with st.expander(
                    f"Passage {index}: {result['source']} "
                    f"({score_label})"
                ):

                    st.write(result["text"])

                    st.caption(
                        "Similarity scores measure retrieval relevance, "
                        "not factual accuracy."
                    )


# --------------------------------------------------
# EMPTY KNOWLEDGE BASE MESSAGE
# --------------------------------------------------

else:

    st.info(
        "Upload and process documents to activate the knowledge base."
    )


# --------------------------------------------------
# APPLICATION FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Business Knowledge Base Assistant | Python, Streamlit, "
    "Gemini API, Sentence Transformers, PyPDF, "
    "python-docx, and scikit-learn"
)

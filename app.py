#!/usr/bin/env python3
#!pip install PyPDF2 pandas tqdm openai -q

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple
import concurrent
import PyPDF2
import os
import pandas as pd
import base64
import re

# Set your OpenAI API key. It's recommended to use Colab secrets for this.
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

# Corrected dir_pdfs to be a directory path where PDFs are stored.
# If you have PDFs in a different directory, please update this path.
#dir_pdfs = '/content/drive/MyDrive/RAG/'
dir_pdfs=r'C:/Users/user.name/Documents/.../rag_pdfs/'

# Ensure the path is a directory and then list only PDF files.
if os.path.isdir(dir_pdfs):
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs) if f.endswith('.pdf')]
else:
    # If it was intended to be a single file, handle it as such
    # For this fix, assuming directory usage, so raising an error if not a directory
    raise ValueError(f"'{dir_pdfs}' is not a directory. Please provide a valid directory path for PDF files.")

#print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

def upload_single_pdf(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
    stats = {"total_files": len(pdf_files), "successful_uploads": 0, "failed_uploads": 0, "errors": []}
    
    #print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

if client and os.getenv("AUTO_INDEX_ON_START", "0") == "1":
    store_name = "streamlit_law_notes_store"
    vector_store_details = create_vector_store(store_name)
    upload_pdf_files_to_vector_store(vector_store_details["id"])
# --------------------------------------------------

# System prompt for Law Notes RAG
SYSTEM_PROMPT = """
You are LawNotes-RAG, an assistant that answers questions using a collection of PDF lecture notes and handouts.

You must answer by retrieving and synthesizing information from these PDFs via the `file_search` tool.

1. Tool usage / RAG:
- You have access to a tool called `file_search` that searches across all uploaded PDFs and returns relevant text chunks.
- Treat this tool as if it were `pdf_search(query, k)` conceptually.
- You MUST call `file_search` before answering whenever:
  - The user asks for definitions, rules, duties, covenants, elements, etc.
  - The question clearly relates to content in the PDFs (landlord & tenant, agency, negligence, etc.).
  - The question is exam-style (explain, discuss, compare, apply).

You may answer without `file_search` only when:
- The user asks about how you work or about the system itself.
- The user asks for generic study advice not tied to the PDF content.
- The user asks you to rephrase or summarise text already provided in the conversation.

You are LawNotes-RAG, an assistant that answers questions using a collection of PDF lecture notes and handouts for law, including topics such as:

- Landlord and Tenant
- Agency
- Negligence
- Related exam and tutorial materials

Your primary goal is to give exam-oriented, accurate, and well-structured answers by retrieving and synthesizing information from these PDFs via the `pdf_search` tool whenever needed.

You MUST strictly follow all instructions below.

==================================================
1. KNOWLEDGE & TRUTHFULNESS
==================================================

1. You do not have general legal knowledge beyond your base model and the provided PDFs. Where there is any conflict between your prior knowledge and the PDFs, the PDFs override for exam-style questions.
2. If the PDFs do not clearly answer the question, you must say so explicitly and MUST NOT fabricate rules, cases, or examples.
   - In that situation, clearly distinguish:
     a) What comes from the notes, and
     b) Your best-effort general reasoning beyond the notes.
   - Example wording:
     “The uploaded notes do not clearly state X. Based on general reasoning, here is my best attempt, which may go beyond the notes…”

==================================================
2. TOOL: file_search(query, k) – WHEN AND HOW TO USE IT
==================================================

You have access to a single tool:

- file_search(query, k)
  - Searches ALL PDFs in the collection.
  - Returns up to k text chunks with metadata like { text, source_title, page_number, score }.

--------------------------
2.1 WHEN YOU MUST USE file_search
--------------------------

You MUST call `file_search` BEFORE answering whenever:

1. The user asks for:
   - Definitions, elements, rules, tests, principles.
   - Lists of duties, rights, remedies, covenants, obligations.
   - Case names or specific statutory references mentioned in the notes.
2. The question is about a specific topic covered by the notes, e.g.:
   - Tenancy types, landlord/tenant duties, covenants.
   - Creation of agency, authority of agent, termination of agency.
   - Negligence, duty of care, breach, causation, defences.
3. The user phrases the request like:
   - “According to the notes…”
   - “From Lesson X…”
   - “What does the handout/lesson say about…”
4. The answer might differ between documents (e.g. overlapping topics across lessons) or you are unsure which lesson it is in.
5. The question is clearly exam-oriented, such as:
   - “Explain…”
   - “Discuss…”
   - “Compare…”
   - “Apply the law to this scenario…”
   - “Advise A and B…”

If you are in doubt whether to use `pdf_search`, YOU MUST use it.

--------------------------
2.2 WHEN YOU MAY ANSWER WITHOUT file_search
--------------------------

You MAY skip `pdf_search` only in these cases:

1. The question is about how YOU work or about the system itself, e.g.:
   - “What tools do you use?”
   - “How do you search the PDFs?”
2. The question is generic learning or study strategy that is not tied to specific content in the PDFs, e.g.:
   - “How should I revise for this exam?”
   - “How can I remember legal cases better?”
3. The user asks you to summarise, simplify, rephrase, or restructure material that is already present in the conversation (for example, the user pasted text from the notes or you just returned a long quote in your last message).

In ALL other situations related to law content, you must call `file_search`.

==================================================
3. RETRIEVAL & REASONING PROTOCOL
==================================================

Whenever you need the PDFs to answer:

1. IDENTIFY INTENT (silently)
   - Determine if the topic is Landlord & Tenant, Agency, Negligence, or something overlapping.
   - Determine whether the user wants:
     - A short recap / definition.
     - A detailed explanation.
     - Scenario/problem analysis with application.

2. CONSTRUCT SEARCH QUERIES
   - Use key doctrinal terms from the question (e.g. “implied covenants of landlord”, “agency by estoppel”, “duty of care negligence”).
   - Include obvious synonyms and related phrases.
   - Choose k based on breadth:
     - Narrow definition-style question → k = 4–6
     - Broad “explain/discuss/compare” → k = 8–10

   Example:
   - If the user asks: “Explain tenancy at will”, you might call:
     pdf_search(query = "tenancy at will definition landlord and tenant", k = 6)

3. ANALYSE RESULTS
   - Prioritise chunks that:
     - Provide definitions, lists, or structured outlines.
     - Contain examples or applications.
     - Look like slides with key points or exam notes.
   - Discard chunks that:
     - Are purely administrative (e.g. “no recording”, title slides).
     - Are clearly unrelated to the user’s question.
     - Are duplicates of better-formulated passages.

4. SYNTHESISE (DO NOT DUMP)
   - Combine relevant chunks into a coherent explanation.
   - Preserve:
     - Element lists.
     - Key terms used in the notes that are likely exam language.
   - Rephrase lightly for clarity, without changing legal meaning.
   - You may indicate approximate lesson or section when helpful (e.g. “In the notes on landlord and tenant…”), but do not fabricate exact page numbers if they are not provided.

==================================================
4. CONCISION VS COMPLETENESS – EXPLICIT TRADE-OFFS
==================================================

You must explicitly manage the trade-off between brevity and coverage.

--------------------------
4.1 DEFAULT MODE: CONCISE BUT COMPLETE-ENOUGH
--------------------------

- Aim for 1–3 short paragraphs OR 3–7 bullet points.
- Cover all core elements needed for a solid exam answer.
- Avoid unnecessary digressions, long quotes, or detailed case histories unless specifically requested.

--------------------------
4.2 PRIORITISE CONCISION WHEN:
--------------------------

- The user says “briefly”, “in one sentence”, “short answer”, “summary”, “flashcard”, etc.
- The question is a straightforward definition or list, e.g. “What is tenancy at will?”
- The user is quickly revising multiple topics.

In these cases:
- Give one clear definition plus at most 1–2 key points.
- Optionally add:
  “Ask me for a full explanation if you want more detail.”

--------------------------
4.3 PRIORITISE COMPLETENESS WHEN:
--------------------------

- The user says “explain in detail”, “full notes”, “teach me”, “step-by-step”, etc.
- The question is an exam-style problem that requires application to facts.
- The topic is central and complex (e.g. creation of agency, types of authority, negligence analysis).

In these cases:
- Structure the answer with clear headings such as:
  - Definition
  - Elements / Requirements
  - Explanation / Examples
  - Application to Scenario / Exam Tips
- Be thorough, but still keep the structure readable and organised.

--------------------------
4.4 IF USER DOES NOT SPECIFY
--------------------------

- Use concise mode for definition/list questions.
- Use more complete mode for “explain/discuss/compare/apply” questions.

Whenever you switch to a more detailed style, SIGNAL this briefly, e.g.:
- “Here is a more detailed explanation: …”

==================================================
5. ANSWER FORMATTING
==================================================

1. Use clear headings and bullet points to mirror how exam notes are structured.
2. For scenario questions:
   - Follow a light IRAC structure (Issue, Rule, Application, Conclusion), even if you do not label the steps explicitly.
   - Base the “Rule” section on retrieved content from `file_search`.
3. Make your answers exam-friendly:
   - Highlight key phrases.
   - Make element lists clear and numbered where appropriate.

==================================================
6. HANDLING UNCERTAINTY & OUT-OF-SCOPE
==================================================

1. If you cannot find a clear answer in the PDFs even after one or two well-constructed `file_search` calls:
   - State clearly that the notes do not cover the point directly.
   - Optionally provide a general explanation based on your broader knowledge, clearly labelled as such.
2. If the question is unrelated to the PDFs (e.g. cooking, general life advice):
   - You may answer using your general knowledge, but clearly indicate this is not derived from the notes.
3. Never fabricate:
   - Cases, statutes, page numbers, or specific quoted wording.
   - If you are unsure, admit uncertainty.

==================================================
7. STYLE & SAFETY
==================================================

1. Be neutral, professional, and supportive.
2. Frame your guidance as help for understanding and exam preparation, not real-world legal advice.
3. If the user appears to need real-world legal help, gently encourage them to consult a qualified lawyer.
- Be clear, structured, and exam-friendly.
- Use bullet points and headings.
- Frame your help as study/exam preparation, not real-world legal advice.
- If the user appears to need real-world legal help, gently encourage them to consult a qualified lawyer.

==================================================
END OF SYSTEM PROMPT.
"""
# --------------------------------------------------
# Streamlit app for Law Notes RAG

from openai import OpenAI
import streamlit as st
import concurrent.futures
import os

# --------------------------------------------------
# Your existing SYSTEM_PROMPT – keep as-is above this
# (not repeated here for brevity)
# --------------------------------------------------
# SYSTEM_PROMPT = """ ... your long LawNotes-RAG system prompt ... """
SYSTEM_PROMPT=SYSTEM_PROMPT

# --------------------------------------------------
# Helpers for vector store + uploads + Q&A
# --------------------------------------------------

@st.cache_resource
def create_vector_store(name: str = "streamlit_law_notes_store", api_key: str = "") -> str:
    """
    Create a fresh vector store for this Streamlit session.
    Returns the vector_store_id.
    """
    if not api_key:
        raise RuntimeError("Missing API key for vector store creation.")
    session_client = OpenAI(api_key=api_key)
    vs = session_client.vector_stores.create(name=name)
    return vs.id


def upload_pdf_to_store(file, vector_store_id: str) -> dict:
    """
    Upload a single Streamlit UploadedFile to OpenAI and attach it to the vector store.
    """
    if client is None:
        return {"file": file.name, "status": "failed", "error": "Missing OpenAI API key."}
    try:
        # file is a Streamlit UploadedFile
        file_obj = client.files.create(
            file=(file.name, file.read()),
            purpose="assistants",
        )
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_obj.id,
        )
        return {"file": file.name, "status": "success"}
    except Exception as e:
        return {"file": file.name, "status": "failed", "error": str(e)}


def _collect_file_metadata(resp: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping of file_id to metadata dictionaries extracted from file_search results.
    """
    metadata: Dict[str, List[Dict[str, Any]]] = {}
    for output in getattr(resp, "output", []):
        if getattr(output, "type", None) != "file_search_call":
            continue
        results = getattr(output, "results", None) or []
        for result in results:
            file_id = getattr(result, "file_id", None)
            if not file_id:
                continue
            metadata.setdefault(file_id, []).append(
                {
                    "filename": getattr(result, "filename", None),
                    "attributes": getattr(result, "attributes", {}) or {},
                }
            )
    return metadata


def _extract_annotation_citations(resp: Any) -> List[str]:
    """
    Convert annotations on response text into formatted citation strings.
    """
    citations: List[str] = []
    metadata = _collect_file_metadata(resp)
    seen_keys: Set[Tuple[str, Any, Any]] = set()

    for output in getattr(resp, "output", []):
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", []):
            if getattr(content, "type", None) != "output_text":
                continue
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "file_citation":
                    continue
                entry = _format_citation_entry(annotation, metadata, seen_keys)
                if entry:
                    citations.append(entry)
    return citations


def _format_citation_entry(
    annotation: Any,
    metadata: Dict[str, List[Dict[str, Any]]],
    seen_keys: Set[Tuple[str, Any, Any]],
) -> str:
    """
    Build a human readable citation label including page / paragraph when available.
    """
    file_id = getattr(annotation, "file_id", None)
    filename = getattr(annotation, "filename", "Unknown source")
    page_value: Any = None
    paragraph_value: Any = None

    for entry in metadata.get(file_id, []):
        filename = entry.get("filename") or filename
        attrs = entry.get("attributes") or {}
        page_value = _get_first(attrs, ("page_number", "page", "page_index", "pageNum"))
        paragraph_value = _get_first(attrs, ("paragraph", "para", "section"))
        if page_value is not None or paragraph_value is not None:
            break

    key = (file_id, page_value, paragraph_value)
    if key in seen_keys:
        return ""
    seen_keys.add(key)

    details = []
    if page_value is not None:
        details.append(f"p. {page_value}")
    if paragraph_value is not None:
        details.append(f"para {paragraph_value}")

    detail_block = f" ({', '.join(details)})" if details else ""
    return f"{filename}{detail_block}".strip()


def _get_first(data: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", []):
            return value
    return None


def _format_response_text(resp: Any) -> str:
    """
    Extract readable text from the Responses object and append citations.
    """
    text_output = ""
    if hasattr(resp, "output_text"):
        try:
            text_output = resp.output_text()
        except Exception:
            text_output = ""

    if not text_output:
        text_output = _fallback_response_text(resp)

    clean_text = re.sub(r"\r\n", "\n", text_output or "").strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    if not clean_text:
        clean_text = "Unable to parse model response."

    citations = _extract_annotation_citations(resp)
    if citations:
        citation_lines = "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(citations))
        clean_text = f"{clean_text}\n\nSources:\n{citation_lines}"

    return clean_text


def _fallback_response_text(resp: Any) -> str:
    """
    Fall back to the first output_text block if Response.output_text() is empty.
    """
    for output in getattr(resp, "output", []):
        if getattr(output, "type", None) == "message":
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "output_text":
                    return getattr(content, "text", "") or ""
    return ""


def ask_question(query: str, vector_store_id: str) -> str:
    """
    Call the Responses API with file_search over the given vector store,
    using your SYSTEM_PROMPT. Returns plain answer text.
    """
    if client is None:
        return "Missing OpenAI API key. Please add it in the sidebar."
    resp = client.responses.create(
        model="gpt-5.1",
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": query}
                ],
            }
        ],
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )

    try:
        return _format_response_text(resp)
    except Exception:
        return str(resp)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="PDF RAG", page_icon="⚖️")

st.title("⚖️ QnA with lecture Notes  – PDF Q&A")

st.markdown(
    """
Upload your **lecture notes / handouts (PDF)** and then ask questions.

The assistant will answer **using only the uploaded PDFs** via RAG.
"""
)

# Create / reuse one vector store for this session
if "indexed" not in st.session_state:
    st.session_state.indexed = False

sidebar_default_key = st.session_state.get("active_api_key") or os.getenv("OPENAI_API_KEY", "")
sidebar_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=sidebar_default_key,
    type="password",
    placeholder="YOUR_OPENAI_API_KEY",
)
sidebar_api_key = sidebar_api_key.strip()
api_key_available = bool(sidebar_api_key)
if api_key_available:
    if sidebar_api_key != os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = sidebar_api_key
    if sidebar_api_key != st.session_state.get("active_api_key"):
        st.session_state["active_api_key"] = sidebar_api_key
        st.session_state.pop("vector_store_id", None)
        st.session_state.indexed = False
    client = OpenAI(api_key=sidebar_api_key)
else:
    st.sidebar.warning("Enter your API key to upload and search your PDFs.")
    st.session_state["active_api_key"] = ""
    client = None

vector_store_id = None
if api_key_available:
    if "vector_store_id" not in st.session_state:
        st.session_state.vector_store_id = create_vector_store(api_key=sidebar_api_key)
    vector_store_id = st.session_state.vector_store_id
else:
    st.session_state.indexed = False

# ---- 1. Upload & Index PDFs ----
st.header("1. Upload your PDF notes")

uploaded_files = st.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    index_disabled = not api_key_available
    if st.button("Index PDFs", disabled=index_disabled):
        if not api_key_available or not vector_store_id:
            st.warning("Add your OpenAI API key before indexing files.")
        else:
            with st.spinner("Uploading and indexing PDFs..."):
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(upload_pdf_to_store, f, vector_store_id)
                    for f in uploaded_files
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        success = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        if success:
            st.session_state.indexed = True
            st.success(f"Indexed {len(success)} file(s).")
            with st.expander("Indexed files"):
                for r in success:
                    st.write("✅", r["file"])

        if failed:
            st.error(f"Failed to index {len(failed)} file(s).")
            with st.expander("Failures"):
                for r in failed:
                    st.write("❌", r["file"], "–", r["error"])

elif not api_key_available:
    st.info("Enter your OpenAI API key in the sidebar to enable indexing.")


# ---- 2. Ask Questions ----
st.header("2. Ask a question about your PDFs")

user_query = st.text_area(
    "Your question",
    height=80,
    placeholder="e.g. Distinguish between professional negligence and negligent misrepresentation",
)

if st.button("Get answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif not api_key_available or not vector_store_id:
        st.warning("Add your OpenAI API key before asking questions.")
    elif not st.session_state.indexed:
        st.warning("Please upload and index at least one PDF first.")
    else:
        with st.spinner("Thinking with your PDFs..."):
            answer = ask_question(user_query.strip(), vector_store_id)

        st.subheader("Answer")
        st.write(answer)


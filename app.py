# app.py
import os
import tempfile
import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from utils import (
    extract_text_from_pdf,
    chunk_text,
    compute_tfidf_similarity,
    get_top_missing_keywords,
    create_md_report,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer — Streamlit + LangChain + Chroma + OpenAI")

# Config / secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit secrets or env variables.")
    st.stop()

# Initialize embeddings and LLM
st.write("Loaded API key:", bool(OPENAI_API_KEY))
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

st.sidebar.header("Options")
save_to_chroma = st.sidebar.checkbox("Store resume in local Chroma DB (faster semantic search)", value=True)
chroma_persist_dir = st.sidebar.text_input("Chroma persist dir", value="./chroma_db")

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use**:\n1. Upload a resume (PDF).\n2. Paste the target Job Description (JD).\n3. Click Analyze.")

with st.form("analyze_form"):
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    uploaded_jd = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
    jd_text_area = st.text_area("Or paste Job Description here (overrides uploaded JD)", height=200)
    submit_btn = st.form_submit_button("Analyze Resume")

if submit_btn:
    if not uploaded_resume:
        st.error("Please upload a resume PDF.")
        st.stop()

    # Extract resume text
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf(uploaded_resume)
    if not resume_text.strip():
        st.error("Could not extract text from the resume PDF. Try a different file or make sure text is selectable.")
        st.stop()

    # Get job description text
    if jd_text_area.strip():
        jd_text = jd_text_area
    elif uploaded_jd:
        if uploaded_jd.type == "text/plain":
            jd_text = uploaded_jd.getvalue().decode("utf-8")
        else:
            jd_text = extract_text_from_pdf(uploaded_jd)
    else:
        st.error("Please paste or upload a Job Description.")
        st.stop()

    st.subheader("Preview — Extracted Text")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resume (first 800 chars)**")
        st.write(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))
    with col2:
        st.markdown("**Job Description (first 800 chars)**")
        st.write(jd_text[:800] + ("..." if len(jd_text) > 800 else ""))

    # --- Quantitative similarity using TF-IDF ---
    with st.spinner("Computing TF-IDF similarity..."):
        tfidf_score = compute_tfidf_similarity(jd_text, resume_text)
    st.subheader("🔹 Quantitative Match")
    st.metric("TF-IDF Similarity (%)", f"{round(tfidf_score * 100, 2)}%")

    # --- Embedding-based semantic search (Chroma) ---
    st.subheader("🔎 Semantic Matching (Embeddings + Chroma)")
    # chunk resume and JD
    resume_chunks = chunk_text(resume_text, chunk_size=800, overlap=120)
    jd_chunks = chunk_text(jd_text, chunk_size=800, overlap=120)

    # Build / load Chroma vectorstore for resume
    vectorstore = None
    if save_to_chroma:
        persist_dir = chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        # Convert to LangChain Document objects
        docs = [Document(page_content=c, metadata={"source": f"resume_chunk_{i}"}) for i, c in enumerate(resume_chunks)]
        try:
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
            vectorstore.persist()
            st.success(f"Chroma persisted resume embeddings to {persist_dir}")
        except Exception as e:
            st.warning(f"Chroma init failed: {e}. Falling back to in-memory embeddings.")
            vectorstore = None

    # If vectorstore is available, query JD against it for semantic match
    semantic_score = None
    top_matches = []
    if vectorstore:
        # create query embeddings for each JD chunk and query
        all_scores = []
        for chunk in jd_chunks:
            results = vectorstore.similarity_search_with_score(chunk, k=3)
            # results: list of (Document, score)
            # Turn score into similarity-like number (smaller = better depending on implementation)
            # We'll average an inverted score (this depends on the vectorstore's score semantics)
            # For display, we will just capture top matched texts
            for doc, score in results:
                all_scores.append((doc.page_content, float(score)))
        # sort by score
        all_scores_sorted = sorted(all_scores, key=lambda x: x[1])
        # take top 5 unique matches
        seen = set()
        for text, score in all_scores_sorted:
            snippet = text[:250]
            if snippet not in seen:
                top_matches.append((snippet, score))
                seen.add(snippet)
            if len(top_matches) >= 5:
                break
        st.write("Top semantic matches (snippets):")
        for i, (txt, score) in enumerate(top_matches, 1):
            st.write(f"**Match {i}** — score: {score:.4f}")
            st.write(txt)
        # crude semantic_score: inversely related to average score
        if all_scores:
            avg_score = sum([s for _, s in all_scores]) / len(all_scores)
            # map avg_score to a 0..1 scale heuristically
            semantic_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_score)))
            st.metric("Semantic Match (heuristic)", f"{round(semantic_score * 100, 2)}%")
    else:
        st.info("Semantic search skipped (Chroma not available).")

    # --- GPT-driven qualitative feedback ---
    st.subheader("🧠 GPT-driven Analysis")
    with st.spinner("Generating AI analysis (strengths, missing keywords, suggestions)..."):
        prompt = f"""
You are an expert technical recruiter and resume coach for data & AI roles.
Given the job description and the resume text, produce JSON with:
1) strengths: bullet list of 4 strengths shown in the resume.
2) missing_skills: bullet list of up to 10 skills/keywords the resume is missing relative to the JD.
3) rewrite_suggestions: 5 concise suggestions to improve the resume for this JD.
4) quick_improvements: 3 action items that the candidate can do in 7 days to boost match.

Respond only with valid JSON keys: strengths, missing_skills, rewrite_suggestions, quick_improvements.
Job Description: {jd_text}
Resume: {resume_text}
"""
        try:
            response = llm.generate([{"role": "user", "content": prompt}])
            # LangChain ChatOpenAI generate returns a complex object; fallback to simple text extraction
            # We'll try to access the content
            ai_text = ""
            try:
                ai_text = response.generations[0][0].text
            except Exception:
                # fallback using chat completion
                ai_text = llm.predict(prompt)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            ai_text = ""

    if ai_text:
        st.code(ai_text, language="json")
    else:
        st.warning("No AI feedback available.")

    # --- Keyword gap (TF-IDF top terms) ---
    st.subheader("⚙️ Keyword Gap (TF-IDF top terms)")
    top_missing = get_top_missing_keywords(jd_text, resume_text, top_n=10)
    if top_missing:
        st.write("Top missing keywords from JD (suggest adding if relevant):")
        st.write(", ".join(top_missing))
    else:
        st.write("No significant missing keywords detected.")

    # --- Generate & download report ---
    report_md = create_md_report(
        resume_text=resume_text,
        jd_text=jd_text,
        tfidf_score=tfidf_score,
        semantic_score=semantic_score,
        ai_feedback=ai_text,
        top_missing=top_missing,
    )
    st.subheader("📥 Downloadable Report")
    st.download_button(
        label="Download report (Markdown)",
        data=report_md,
        file_name="resume_analysis_report.md",
        mime="text/markdown",
    )

    st.success("Analysis complete! Tweak inputs and rerun as needed.")

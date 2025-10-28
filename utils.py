# utils.py
import io
import re
import pdfplumber
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# -----------------------------
# 1️⃣ Robust PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract readable UTF-8 text from a PDF using pdfplumber (handles encoding better than PyPDF2).
    Returns cleaned text suitable for NLP processing.
    """
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        # Normalize encoding and clean stray symbols
        text = text.encode("utf-8", "ignore").decode("utf-8")
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII garbage
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        text = f"Error extracting PDF text: {e}"
    return text


# -----------------------------
# 2️⃣ Text Chunking for Embeddings
# -----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks for embedding-based similarity."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    # Add slight overlap to preserve context
    if overlap > 0 and len(chunks) > 1:
        merged = []
        for i, c in enumerate(chunks):
            if i == 0:
                merged.append(c)
            else:
                prev = merged[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                merged.append((overlap_text + " " + c).strip())
        return merged
    return chunks


# -----------------------------
# 3️⃣ TF-IDF Similarity Computation
# -----------------------------
def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity between two documents using TF-IDF (0–1 range)."""
    if not text_a or not text_b:
        return 0.0
    vect = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vect.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(score)
    except Exception:
        return 0.0


# -----------------------------
# 4️⃣ Identify Missing Keywords
# -----------------------------
def get_top_missing_keywords(jd_text: str, resume_text: str, top_n: int = 10):
    """Find top keywords present in JD but missing from resume."""
    vect = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        docs = [jd_text, resume_text]
        tfidf = vect.fit_transform(docs)
        feature_names = vect.get_feature_names_out()
        jd_vec = tfidf[0].toarray()[0]
        resume_vec = tfidf[1].toarray()[0]
        diffs = [(feature_names[i], jd_vec[i] - resume_vec[i]) for i in range(len(feature_names))]
        diffs_sorted = sorted(diffs, key=lambda x: x[1], reverse=True)
        missing = [w for w, diff in diffs_sorted if diff > 0][:top_n]
        return missing
    except Exception:
        return []


# -----------------------------
# 5️⃣ Markdown Report Generator
# -----------------------------
def create_md_report(resume_text, jd_text, tfidf_score, semantic_score, ai_feedback, top_missing):
    """Create a markdown report summarizing results for easy export."""
    md = []
    md.append("# Resume Analysis Report\n")

    # Scores
    md.append("## Scores")
    md.append(f"- **TF-IDF Similarity:** {round(tfidf_score * 100, 2)}%")
    if semantic_score is not None:
        md.append(f"- **Semantic Match (heuristic):** {round(semantic_score * 100, 2)}%")
    md.append("")

    # Missing keywords
    md.append("## Top Missing Keywords")
    if top_missing:
        md.append(", ".join(top_missing))
    else:
        md.append("_No significant keywords missing._")
    md.append("")

    # AI feedback
    md.append("## AI Feedback (raw JSON/text)")
    md.append("```json")
    try:
        obj = json.loads(ai_feedback)
        md.append(json.dumps(obj, indent=2))
    except Exception:
        md.append(ai_feedback)
    md.append("```")

    # Content previews
    md.append("\n## Job Description (excerpt)")
    md.append(jd_text[:1500] + ("..." if len(jd_text) > 1500 else ""))
    md.append("\n## Resume (excerpt)")
    md.append(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))
    return "\n".join(md)

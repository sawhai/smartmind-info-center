#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Configuration - UPDATED FOR DEPLOYMENT
# ---------------------------
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# FIXED: Use relative path for deployment
DOCX_PATH = PROJECT_ROOT / "docs" / "smartmind_docs_v2.docx"

META_PATH = INDEX_DIR / "meta.jsonl"
EMB_PATH  = INDEX_DIR / "embeddings.npy"

# Load environment variables - UPDATED for deployment
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL_ENV = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini").strip()

# Constants
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
TOP_K = 5
COVERAGE_THRESHOLD = 0.22
AVG_THRESHOLD = 0.18
MAX_TOKENS = 1200
TEMPERATURE = 0.2
TOP_P = 1.0

# ---------------------------
# Streamlit Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Powered Smart Mind Info Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
/* Hide Streamlit elements */
.stDeployButton {display:none;}
footer {visibility: hidden;}
.stApp > header {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* RTL Support */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}

/* Container styling */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1000px;
    margin: 0 auto;
}

/* Header styling */
.main-header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2.5rem 1rem;
    background: linear-gradient(135deg, #f8fffe 0%, #e8f4f2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #2c5530;
    margin: 1rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    letter-spacing: 1px;
}

.logo-image {
    width: 120px;
    height: auto;
    margin-bottom: 1rem;
}

/* Tips section */
.tips-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f9ff 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 2rem 0;
    border-right: 5px solid #2c5530;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.tips-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #2c5530;
    margin-bottom: 1rem;
}

.tips-content {
    font-size: 1rem;
    color: #444;
    line-height: 1.6;
}

/* Chat section */
.chat-title {
    font-size: 2rem;
    font-weight: bold;
    color: #2c5530;
    text-align: center;
    margin: 2rem 0 1rem 0;
}

.chat-container {
    max-height: 500px;
    overflow-y: auto;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: #fafafa;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
}

/* Chat bubbles - both on right side with icons */
.chat-bubble {
    max-width: 80%;
    margin: 1rem 0;
    padding: 1.2rem 1.5rem;
    border-radius: 20px;
    float: right;
    clear: both;
    font-size: 1.1rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    word-wrap: break-word;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.user-bubble {
    background: linear-gradient(135deg, #b0b0b0, #999);
    color: white;
    border-radius: 20px 20px 5px 20px;
    margin-left: 15%;
}

.smartmind-bubble {
    background: linear-gradient(135deg, #66bb6a, #4caf50);
    color: white;
    border-radius: 20px 20px 5px 20px;
    margin-left: 10%;
    margin-top: 0.5rem;
}

.chat-icon {
    font-size: 1.5rem;
    min-width: 1.5rem;
    margin-top: 2px;
    opacity: 0.9;
}

.chat-content {
    flex: 1;
}

.bubble-label {
    font-size: 0.9rem;
    font-weight: bold;
    margin-bottom: 0.8rem;
    opacity: 0.9;
}

.chat-clear {
    clear: both;
    height: 1rem;
}

/* Input styling */
.input-section {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #e0e0e0;
    margin-bottom: 2rem;
}

.stTextInput > div > div > input {
    border-radius: 25px !important;
    padding: 1rem 1.5rem !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 1.1rem !important;
    font-family: 'Arial', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4caf50, #66bb6a) !important;
    color: white !important;
    border-radius: 25px !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3) !important;
}

/* Custom spinner */
.custom-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    margin: 1rem 0;
}

.spinner-container {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.spinner-text {
    font-size: 1.1rem;
    color: #2c5530;
    font-weight: 500;
}

.spinner-dots {
    display: flex;
    gap: 0.5rem;
}

.spinner-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #2c5530;
    animation: bounce 1.4s ease-in-out infinite both;
}

.spinner-dot:nth-child(1) { animation-delay: -0.32s; }
.spinner-dot:nth-child(2) { animation-delay: -0.16s; }
.spinner-dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #666;
    background: #fafafa;
    border-radius: 15px;
    margin: 2rem 0;
}

.empty-state h3 {
    color: #2c5530;
    font-size: 1.8rem;
    margin-bottom: 1rem;
}

/* Setup warning */
.setup-warning {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 2rem 0;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Helper Functions
# ---------------------------
def normalize_ar(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u0640]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()

@st.cache_data(show_spinner=False)
def read_docx_to_pages(docx_path: Path) -> List[Dict]:
    title = docx_path.name
    doc = Document(str(docx_path))
    pages, buf, page_no = [], [], 1

    def flush():
        nonlocal buf, page_no
        text = "\n".join(buf).strip()
        if text:
            pages.append({
                "doc_id": docx_path.stem,
                "title": title,
                "page_number": page_no,
                "text": normalize_ar(text)
            })
        buf = []
        page_no += 1

    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if not t:
            if buf: flush()
            continue
        buf.append(t)
    if buf: flush()
    return pages

def chunk_pages(pages: List[Dict], target_chars=1800, overlap_chars=250) -> List[Dict]:
    chunks = []
    for pg in pages:
        raw = pg["text"]
        segments = re.split(r"(?:\n\n+|•|- |\u2022|\u25CF|•|\.|؛|:)\s*", raw)
        segments = [s.strip() for s in segments if s and s.strip()]
        buf = ""
        for seg in segments:
            if not buf:
                buf = seg
                continue
            candidate = (buf + " " + seg).strip()
            if len(candidate) <= target_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append({
                        "doc_id": pg["doc_id"], "title": pg["title"], "page_number": pg["page_number"],
                        "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                        "text": buf
                    })
                overlap = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
                buf = (overlap + " " + seg).strip()
        if buf:
            chunks.append({
                "doc_id": pg["doc_id"], "title": pg["title"], "page_number": pg["page_number"],
                "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                "text": buf
            })
    return chunks

def embed_texts_openai(texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    E = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    bs, i = 64, 0
    while i < len(texts):
        batch = texts[i:i+bs]
        clean = [(t or " ").replace("\n", " ").strip() for t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=clean)
        for j, e in enumerate(resp.data):
            E[i+j] = np.array(e.embedding, dtype="float32")
        i += bs
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return E / norms

def save_index(chunks: List[Dict], E: np.ndarray):
    np.save(EMB_PATH, E)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[NearestNeighbors, np.ndarray, List[str]]:
    E = np.load(EMB_PATH)
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(E)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_lines = f.readlines()
    return nn, E, meta_lines

def embed_query(q: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    r = client.embeddings.create(model=EMBED_MODEL, input=[q.replace("\n", " ").strip()])
    v = np.array(r.data[0].embedding, dtype="float32")
    n = np.linalg.norm(v) or 1.0
    return v / n

def build_sources(meta_lines: List[str], idxs: np.ndarray, trim=900) -> str:
    out_lines = []
    for i, idx in enumerate(idxs.tolist(), start=1):
        m = json.loads(meta_lines[int(idx)])
        body = m["text"].replace("\n", " ").strip()
        if len(body) > trim: 
            body = body[:trim] + "..."
        header = f"{i}) [العنوان: {m['title']} | ص.{m['page_number']} | Chunk: {m['chunk_id']}]"
        out_lines.append(header + "\n" + body)
    return "\n\n".join(out_lines)

def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI as DeepSeekClient
    base = DEEPSEEK_BASE_URL.rstrip("/")
    bases = [base, base + "/v1"] if not base.endswith("/v1") else [base, base[:-3]]
    candidates = ["deepseek-reasoner", "deepseek-chat"]

    for b in bases:
        try:
            ds = DeepSeekClient(api_key=DEEPSEEK_API_KEY, base_url=b)
            for model in candidates:
                try:
                    comp = ds.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS
                    )
                    return comp.choices[0].message.content
                except:
                    continue
        except:
            continue
    raise Exception("DeepSeek failed")

def call_openai_fallback(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    comp = client.chat.completions.create(
        model=OPENAI_FALLBACK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    return comp.choices[0].message.content

def clean_answer(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ---------------------------
# Pre-deployment checks
# ---------------------------
def check_setup():
    issues = []
    
    # Check API keys
    if not OPENAI_API_KEY:
        issues.append("❌ OPENAI_API_KEY not set")
    else:
        issues.append("✅ OpenAI API key configured")
        
    if not DEEPSEEK_API_KEY:
        issues.append("⚠️ DEEPSEEK_API_KEY not set (will use OpenAI fallback)")
    else:
        issues.append("✅ DeepSeek API key configured")
    
    # Check document
    if not DOCX_PATH.exists():
        issues.append(f"❌ Knowledge document not found: {DOCX_PATH}")
        issues.append("   Please place 'smartmind_docs_v2.docx' in the 'docs' folder")
    else:
        issues.append("✅ Knowledge document found")
    
    # Check index
    if not (EMB_PATH.exists() and META_PATH.exists()):
        if DOCX_PATH.exists() and OPENAI_API_KEY:
            issues.append("⚠️ Search index not built yet (will build automatically)")
        else:
            issues.append("❌ Cannot build search index (missing requirements)")
    else:
        issues.append("✅ Search index ready")
    
    return issues

# ---------------------------
# Main App Logic
# ---------------------------

# Check setup first
setup_issues = check_setup()
critical_issues = [issue for issue in setup_issues if issue.startswith("❌")]

if critical_issues:
    st.error("⚠️ Setup Issues Found")
    for issue in setup_issues:
        st.write(issue)
    st.stop()

# Check if index exists and needs to be built
if not (EMB_PATH.exists() and META_PATH.exists()):
    if DOCX_PATH.exists() and OPENAI_API_KEY:
        st.info("🔄 Building search index for the first time. This may take a few minutes...")
        
        try:
            # Build index
            pages = read_docx_to_pages(DOCX_PATH)
            chunks = chunk_pages(pages, target_chars=1800, overlap_chars=250)
            texts = [chunk["text"] for chunk in chunks]
            
            with st.spinner("Creating embeddings..."):
                embeddings = embed_texts_openai(texts)
            
            save_index(chunks, embeddings)
            st.success("✅ Search index built successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to build index: {str(e)}")
            st.stop()
    else:
        st.error("Cannot build search index - missing requirements")
        st.stop()

# Load index
try:
    nn, E, meta_lines = load_index()
except Exception as e:
    st.error(f"❌ Failed to load search index: {str(e)}")
    st.stop()

# ---------------------------
# UI Layout
# ---------------------------

# Header with logo and title
st.markdown('<div class="main-header">', unsafe_allow_html=True)

logo_path = PROJECT_ROOT / "logo.png"
if logo_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(logo_path), width=120)

st.markdown('<h1 class="main-title">AI Powered Smart Mind Info Center</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tips section
st.markdown("""
<div class="tips-box">
    <div class="tips-title">💡 نصائح للحصول على أفضل إجابة</div>
    <div class="tips-content">
        • اكتب سؤالك بوضوح عن خدمة أو سياسة، واذكر التفاصيل المهمة (البرنامج/المستوى/الفرع/التاريخ) لتحصل على إجابة أدق<br>
        • أجيب من المستندات المرفقة فقط؛ وإن لم تتوفر الإجابة، سأقول إنني لم أعثر عليها
    </div>
</div>
""", unsafe_allow_html=True)

# Chat section title
st.markdown('<h2 class="chat-title">💬 اطرح سؤالك</h2>', unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for chat in reversed(st.session_state.chat_history):
        # User bubble with icon
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <div class="chat-icon">👤</div>
            <div class="chat-content">
                <div class="bubble-label">أنت:</div>
                {chat["question"]}
            </div>
        </div>
        <div class="chat-clear"></div>
        """, unsafe_allow_html=True)
        
        # Smart Mind bubble with brain icon
        st.markdown(f"""
        <div class="chat-bubble smartmind-bubble">
            <div class="chat-icon">🧠</div>
            <div class="chat-content">
                <div class="bubble-label">سمارت مايند:</div>
                {chat["answer"]}
            </div>
        </div>
        <div class="chat-clear"></div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_question = st.text_input(
            label="سؤالك",
            placeholder="اكتب سؤالك هنا...",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col2:
        submit_button = st.form_submit_button("➤")

# Process question
if submit_button and user_question.strip():
    # Show loading for retrieval
    spinner1 = st.empty()
    spinner1.markdown("""
    <div class="custom-spinner">
        <div class="spinner-container">
            <div class="spinner-text">جاري البحث في المستندات...</div>
            <div class="spinner-dots">
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieve relevant passages
    q_vec = embed_query(user_question)
    n_neighbors = min(TOP_K, E.shape[0])
    distances, indices = nn.kneighbors(q_vec.reshape(1, -1), n_neighbors=n_neighbors)
    sims = 1.0 - distances[0]
    idxs = indices[0]
    
    spinner1.empty()
    
    best_sim = float(sims[0]) if len(sims) else 0.0
    avg_sim = float(np.mean(sims)) if len(sims) else 0.0
    
    # Check coverage
    if best_sim < COVERAGE_THRESHOLD and avg_sim < AVG_THRESHOLD:
        answer = "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
    elif best_sim < COVERAGE_THRESHOLD:
        answer = "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
    else:
        # Build context
        sources_block = build_sources(meta_lines, idxs, trim=900)
        
        # Prepare prompts
        system_prompt = (
            "أنت مساعد عربي يجيب حصراً من المقاطع المزوّدة. "
            "اكتب إجابة واحدة موجزة ومنظمة. لا تستخدم رموز التنسيق مثل ** أو *. "
            "إذا لم تكفِ المعلومات، قل: لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
        )
        
        user_prompt = (
            f"سؤال المستخدم:\n{user_question}\n\n"
            f"المصادر:\n{sources_block}\n\n"
            "أجب بإيجاز مع ذكر المراجع بالشكل [العنوان، ص.X]"
        )
        
        # Show loading for generation
        spinner2 = st.empty()
        spinner2.markdown("""
        <div class="custom-spinner">
            <div class="spinner-container">
                <div class="spinner-text">جاري تحضير الإجابة...</div>
                <div class="spinner-dots">
                    <div class="spinner-dot"></div>
                    <div class="spinner-dot"></div>
                    <div class="spinner-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate answer
        answer = None
        try:
            if DEEPSEEK_API_KEY:
                answer = call_deepseek(system_prompt, user_prompt)
        except:
            pass
        
        if answer is None:
            answer = call_openai_fallback(system_prompt, user_prompt)
        
        spinner2.empty()
        answer = clean_answer(answer)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": answer
    })
    
    st.rerun()

# Empty state
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <h3>مرحباً بك في مركز معلومات سمارت مايند الذكي</h3>
        <p>اطرح سؤالك أعلاه للحصول على إجابة دقيقة من مستنداتنا</p>
    </div>
    """, unsafe_allow_html=True)
# smartmind_app.py — SmartMind Q&A (nice UI)
# ------------------------------------------------------------
# - Chat-style layout with history
# - Larger, elegant typography + RTL
# - Citation chips (expanders show page text)
# - Works by importing your local app_core.ask / PAGE_INDEX
# ------------------------------------------------------------

import json, re, os
from pathlib import Path
import streamlit as st
from pathlib import Path

# --- Robust logo resolver ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # your app lives here; adjust only if needed

def resolve_logo() -> Path | None:
    # Try common locations relative to this file
    candidates = [
        SCRIPT_DIR / "logo.png",                # same folder as smartmind_app.py
        SCRIPT_DIR.parent / "logo.png",         # parent folder
        SCRIPT_DIR / "chatbot" / "logo.png",    # ./chatbot/logo.png
        SCRIPT_DIR.parent / "chatbot" / "logo.png",
        Path("/Users/ha/My Drive/SmartMind Consultation/Chatbots/logo.png"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

LOGO_PATH = resolve_logo()  

# ====== import your core engine here ======
# make sure app_core.py exposes: ask(question, keywords=list[str]) and PAGE_INDEX (dict[int,str])
from app_core import ask, PAGE_INDEX

# ====== helpers ======
AR_DIAC  = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
def normalize_ar(s: str) -> str:
    if not s: return s
    s = s.replace("\u0640", "")             # tatweel
    s = AR_DIAC.sub("", s)                  # remove diacritics
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_keywords_ar(q: str, k_max=6):
    q = normalize_ar(q or "")
    tokens = re.split(r"[\s،,:;.!؟\-\(\)\[\]\{\}]+", q)
    tokens = [t for t in tokens if t and len(t) > 2]
    uniq = list(dict.fromkeys(tokens))
    return sorted(uniq, key=len, reverse=True)[:k_max] if uniq else []

def preview_page(page: int, n_chars=950) -> str:
    body = PAGE_INDEX.get(page, "")
    if not body: return "[لا يوجد نص في هذه الصفحة]"
    s = body[:n_chars].replace("\n", " ")
    return s + ("…" if len(body) > n_chars else "")

# ====== page config & styles ======
#st.set_page_configst.set_page_config(page_title="Smart Mind Info Center", page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📘", layout="wide")
import streamlit as st

st.set_page_config(
    page_title="Smart Mind Info Center",
    page_icon=(str(LOGO_PATH) if LOGO_PATH else "📘"),
    layout="wide"
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { direction: rtl; }
:root {
  --sm-font: 17px;
  --sm-font-lg: 19px;
  --sm-radius: 16px;
  --sm-border: #E5E7EB;
  --sm-muted: #6B7280;
}
html, body { font-size: var(--sm-font); }
h1,h2,h3,h4 { letter-spacing: -.015em; }

/* Cards */
.sm-card {
  border: 1px solid var(--sm-border);
  background: #FFFFFF;
  border-radius: var(--sm-radius);
  padding: 18px 22px;
  box-shadow: 0 1px 0 rgba(16,24,40,.02);
}

/* Clean professional header (no gradients) */
.header-wrap {
  border: 1px solid var(--sm-border);
  background: #FFFFFF;
  border-radius: 20px;
  padding: 16px 18px;
}

/* Bubbles */
.user-bubble, .bot-bubble {
  border-radius: 18px;
  padding: 16px 18px;
  max-width: 1000px;
}
.user-bubble { background: #F1F5F9; border: 1px solid #E2E8F0; }
.bot-bubble  { background: #F6FFF8; border: 1px solid #CFF4D2; }

/* Inputs */
textarea, input { font-size: var(--sm-font-lg) !important; }
.stTextArea textarea { min-height: 140px !important; }

/* Footer */
.footer { color:#94A3B8; font-size:12px; margin-top:24px; }
</style>
""", unsafe_allow_html=True)
    
# ====== sidebar (about/settings) ======

with st.sidebar:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=110)
    st.markdown("### Smart Mind Info Center")
    st.caption("إجابات موثقة من مستنداتك الداخلية.")
    st.markdown("---")
    st.caption("نصيحة: استخدم كلمات مفتاحية واضحة (مثل: الانسحاب، الألمانية، الرسوم).")
    if st.button("🧹 مسح المحادثة", use_container_width=True):
        st.session_state.pop("chat", None)
        st.rerun()
        
# ====== header ======
# col_a, col_b = st.columns([1, 5])
# with col_a:
#     st.markdown('<div class="header-wrap sm-card" style="width:70px;height:70px;display:grid;place-items:center;font-size:26px;">📘</div>', unsafe_allow_html=True)
# with col_b:
#     st.markdown('<div class="header-wrap sm-card"><h2 style="margin:0">SmartMind Q&A</h2><div class="sm-caption">اسأل عن البرامج، الرسوم، السياسات… وستصلك إجابة مع مرجع الصفحات.</div></div>', unsafe_allow_html=True)
# ==== HEADER: Smart Mind Info Center ====
# ==== HEADER: Smart Mind Info Center ====
# ==== Header: Smart Mind Info Center ====
st.markdown('<div class="header-wrap sm-card">', unsafe_allow_html=True)
cols = st.columns([1, 9])
with cols[0]:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=72)
    else:
        st.write("📘")
with cols[1]:
    st.markdown("### Smart Mind Info Center")
    st.markdown('<div style="font-size:13px;color:#6B7280">منصة استعلام ذكية — إجابات دقيقة مستندة إلى قاعدة المعرفة مع الاستشهاد بالصفحات</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ====== session state for chat ======
if "chat" not in st.session_state:
    st.session_state.chat = []   # list[dict]: {role:"user"|"assistant", "content":str, "citations":list}

# ====== show chat history ======
for turn in st.session_state.chat:
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        bubble_class = "user-bubble" if turn["role"] == "user" else "bot-bubble"
        #st.markdown(f'<div class="{bubble_class} sm-card">{turn["content"]}</div>', unsafe_allow_html=True)
        if turn["role"] == "assistant":
            # Assistant bubbles may contain HTML (styled answers)
            st.markdown(f'<div class="{bubble_class} sm-card">{turn["content"]}</div>', unsafe_allow_html=True)
        else:
            # User bubbles should be plain text (escape manually)
            safe_text = turn["content"].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(f'<div class="{bubble_class} sm-card">{safe_text}</div>', unsafe_allow_html=True)
        # show citations for assistant
        if turn["role"] == "assistant" and turn.get("citations"):
            st.write("**المراجع:**")
            for c in turn["citations"]:
                page = int(c.get("page") or c.get("PAGE") or 0)
                if not page: 
                    continue
                with st.expander(f"الصفحة {page} — عرض مقتطف"):
                    st.markdown(preview_page(page), unsafe_allow_html=True)

# ====== input area ======
st.write("")  # spacing
with st.container():
    q = st.text_area("سؤالك", placeholder="اكتب سؤالك هنا…", height=150, label_visibility="visible")
    detected = extract_keywords_ar(q)
    st.markdown("**كلمات مفتاحية متوقعة:** " + ("، ".join(detected) if detected else "—"))

col_left, col_right = st.columns([6, 1])
with col_right:
    send = st.button("إرسال", use_container_width=True, disabled=not q.strip())

# ====== handle send ======
if send and q.strip():
    # append user turn
    st.session_state.chat.append({"role":"user", "content": q.strip()})
    with st.spinner("جاري البحث في قاعدة المعرفة…"):
        res = ask(q.strip(), keywords=detected)
    # format assistant turn
    if res.get("status") == "answered":
        ans = res.get("answer") or ""
        cites = res.get("citations") or []
        #st.session_state.chat.append({"role":"assistant", "content": ans, "citations": cites})
        styled_ans = f'<div style="font-size:18px; line-height:1.95">{ans}</div>'
        st.session_state.chat.append({"role":"assistant", "content": styled_ans, "citations": cites})
    else:
        reason = res.get("reason") or res.get("answer") or "لا توجد إجابة مؤكدة في قاعدة المعرفة لهذا السؤال."
        st.session_state.chat.append({"role":"assistant", "content": f"لا توجد إجابة مؤكدة.\n\n**السبب:** {reason}", "citations": []})
    st.rerun()

# ====== footer ======
st.markdown('<div class="footer">© SmartMind — الإجابات مستندة حصراً إلى قاعدة المعرفة لديك مع إظهار صفحات المرجع.</div>', unsafe_allow_html=True)
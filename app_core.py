#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Mind Info Center — Core Engine
------------------------------------
- Loads a prebuilt knowledge pack (kb/v1/knowledge_pack.md) with anchors "== PAGE N ==".
- Answers only from the pack; returns JSON with citations.
- Verifies citations (pages exist + basic numeric sanity).
- Supports DeepSeek or OpenAI via env vars or Streamlit secrets.
"""

# ===== Imports =====
import os, re, json, orjson
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
import requests

# ===== Optional: bridge Streamlit Secrets -> os.environ =====
try:
    import streamlit as st  # type: ignore
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

# ===== Paths (relative & cloud-safe) =====
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
PACK_PATH    = PROJECT_ROOT / "v1" / "knowledge_pack.md"  # required at deploy
LOGO_PATH    = PROJECT_ROOT / "logo.png"  # optional (used by UI)

# ===== Provider selection (DeepSeek preferred if base_url provided) =====
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = (os.getenv("DEEPSEEK_BASE_URL") or "").rstrip("/")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY:
    PROVIDER = "deepseek"
    API_KEY  = DEEPSEEK_API_KEY
    BASE_URL = f"{DEEPSEEK_BASE_URL}/v1"
    MODEL    = DEEPSEEK_MODEL
elif OPENAI_API_KEY:
    PROVIDER = "openai"
    API_KEY  = OPENAI_API_KEY
    BASE_URL = "https://api.openai.com/v1"
    MODEL    = "gpt-4o-mini"
else:
    raise SystemExit("No API key found. Set DEEPSEEK_* or OPENAI_API_KEY via env or Streamlit secrets.")

# ===== Arabic normalization =====
AR_DIAC  = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")  # diacritics
TATWEEL  = "\u0640"
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(s: str) -> str:
    return (s or "").translate(ARABIC_INDIC)

def normalize_ar(s: str) -> str:
    if not s: return s
    s = s.replace(TATWEEL, "")
    s = AR_DIAC.sub("", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
    s = re.sub(r"ا{2,}", "ا", s)      # collapse repeated alef from OCR
    s = s.replace("اال", "ال")        # common OCR artifact
    s = re.sub(r"\s+", " ", s).strip()
    s = normalize_digits(s)
    return s

# ===== Knowledge pack loader & index =====
def _read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing knowledge pack at: {p}")
    return p.read_text(encoding="utf-8")

KB_TEXT = _read_text(PACK_PATH)

PAGE_RE = re.compile(r"^\s*==\s*PAGE\s+(\d+)\s*==\s*$", re.IGNORECASE | re.MULTILINE)

def build_page_index(pack_text: str) -> Dict[int, str]:
    pages: Dict[int, str] = {}
    matches = list(PAGE_RE.finditer(pack_text))
    for i, m in enumerate(matches):
        pnum  = int(m.group(1))
        start = m.end()
        end   = matches[i+1].start() if i + 1 < len(matches) else len(pack_text)
        pages[pnum] = pack_text[start:end].strip()
    return pages

PAGE_INDEX: Dict[int, str] = build_page_index(KB_TEXT)
ALLOWED_PAGES: List[int]   = sorted(PAGE_INDEX.keys())
if not ALLOWED_PAGES:
    raise RuntimeError("Knowledge pack loaded, but no pages were indexed (no '== PAGE N ==' anchors?).")

# Normalized text for fuzzy hinting
KB_TEXT_NORM = normalize_ar(KB_TEXT)
PAGE_INDEX_NORM: Dict[int, str] = {p: normalize_ar(t) for p, t in PAGE_INDEX.items()}

# ===== Fuzzy page hints =====
def _fuzzy_pages_for(term: str, top_k: int = 10) -> List[Tuple[int, int]]:
    t = normalize_ar(term)
    scores: List[Tuple[int, int]] = []
    for p, txt in PAGE_INDEX_NORM.items():
        s = fuzz.partial_ratio(t, txt)  # 0..100
        if s > 0:
            scores.append((s, p))
    scores.sort(reverse=True)
    return scores[:top_k]  # (score, page)

def find_pages_fuzzy(keywords: Optional[List[str]], top_k: int = 8) -> List[int]:
    if not keywords:
        return []
    seen, bag = set(), []
    for kw in keywords:
        for s, p in _fuzzy_pages_for(kw, top_k=top_k):
            if p not in seen:
                seen.add(p); bag.append(p)
            if len(bag) >= top_k:
                break
    return bag

# ===== System prompt =====
ALLOWED_PAGES_STR = ", ".join(map(str, ALLOWED_PAGES))
SYSTEM_PROMPT_AR = f"""
أنت مساعد خبير لـ Smart Mind. مهمتك الإجابة فقط من "قاعدة المعرفة" المرفقة داخل نفس الرسالة.

قواعد صارمة:
1) لا تستخدم أي معرفة خارجية إطلاقاً.
2) أعد الخرج كـ JSON صالح فقط.
3) إن لم تغطِّ القاعدة السؤال، أعد status="not_covered" مع سبب مختصر.
4) إذا أجبتَ، يجب تضمين استشهاد واحد على الأقل بصيغة {{ "page": N }}.
5) يمنع اختلاق الاستشهادات — يجب أن يكون N ضمن الصفحات المسموح بها فقط.
6) الصفحات المسموح بها: [{ALLOWED_PAGES_STR}].

تنسيق الخرج:
{{
  "status": "answered" | "not_covered",
  "answer": "<نص بالعربية>",
  "citations": [ {{ "page": <رقم صفحة صحيح> }} ],
  "disclaimer": "أجبتُ فقط من قاعدة المعرفة."
}}
"""

def _build_messages(user_q: str, kb_text: str, keywords: Optional[List[str]] = None):
    hints = ""
    focus_excerpt = ""
    pages = find_pages_fuzzy(keywords, top_k=6) if keywords else []
    if pages:
        hints = "صفحات مرشحة: " + ", ".join(map(str, pages)) + "\n"
        snippets = []
        for p in pages:
            t = PAGE_INDEX.get(p, "")
            if t:
                snippets.append(f"== PAGE {p} ==\n{t[:900]}")
        if snippets:
            focus_excerpt = "\n\n[FOCUS]\n" + "\n\n".join(snippets) + "\n[/FOCUS]\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AR.strip()},
        {"role": "user",   "content": f"{hints}سؤال المستخدم:\n{user_q}\n\n---\nقاعدة المعرفة:\n{focus_excerpt}{kb_text}\n---"}
    ]

# ===== Model call =====
def _chat_completion(messages, model, base_url, api_key, temperature=0.0, max_tokens=900) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ===== JSON & verification =====
def _parse_json_safely(text: str) -> dict:
    try:
        return orjson.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            return orjson.loads(m.group(0))
        raise

def _numeric_supported_in_pages(answer_dict: dict, page_index: Dict[int, str]) -> bool:
    if answer_dict.get("status") != "answered":
        return True
    ans  = normalize_digits(answer_dict.get("answer", ""))
    nums = re.findall(r"\d[\d,\.]*", ans)
    if not nums:
        return True
    pages: List[int] = []
    for c in answer_dict.get("citations", []):
        if isinstance(c, dict):
            if "page" in c: pages.append(int(c["page"]))
            elif "PAGE" in c: pages.append(int(c["PAGE"]))
    blob = normalize_digits("\n".join(page_index.get(p, "") for p in pages))
    return all(n in blob for n in nums)

def _verify_citations(answer_dict: dict, page_index: Dict[int, str]) -> Tuple[bool, str]:
    if answer_dict.get("status") == "not_covered":
        return True, ""
    cites = answer_dict.get("citations") or []
    if not cites:
        return False, "No citations."
    pages: List[int] = []
    for c in cites:
        try:
            if "page" in c: pages.append(int(c["page"]))
            elif "PAGE" in c: pages.append(int(c["PAGE"]))
        except Exception:
            pass
    if not pages:
        return False, "Bad citation format."
    if any(p not in page_index for p in pages):
        return False, "Cited page does not exist."
    if not (answer_dict.get("answer") or "").strip():
        return False, "Empty answer."
    if not _numeric_supported_in_pages(answer_dict, page_index):
        return False, "Numeric values not supported by cited pages."
    return True, ""

# ===== Public API =====
def ask(question: str, keywords: Optional[List[str]] = None) -> dict:
    """
    Run a grounded Q&A over the knowledge pack.
    Returns a JSON dict:
      - {"status":"answered","answer": "...","citations":[{"page":N},...],"disclaimer":"..."}
      - or {"status":"not_covered","reason":"...","citations":[]}
    """
    messages = _build_messages(question, KB_TEXT, keywords=keywords)
    raw      = _chat_completion(messages, MODEL, BASE_URL, API_KEY)
    try:
        data = _parse_json_safely(raw)
    except Exception as e:
        return {"status": "not_covered", "reason": f"JSON invalid: {e}", "citations": []}
    ok, reason = _verify_citations(data, PAGE_INDEX)
    if not ok:
        return {"status": "not_covered", "reason": f"تعذّر التحقق: {reason}", "citations": []}
    data.setdefault("disclaimer", "أجبتُ فقط من قاعدة المعرفة.")
    return data

# ===== Optional: local debug helper =====
def preview_pages(pages: List[int], n_chars: int = 400):
    for p in pages:
        body = PAGE_INDEX.get(p, "")
        print(f"\n===== PAGE {p} =====")
        print(body[:n_chars].replace("\n", " ") + ("…" if len(body) > n_chars else ""))
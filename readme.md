# 📘 Smart Mind Info Center

An AI-powered Q&A prototype for **Smart Mind Educational Group**.  
This app answers user questions **only from the internal knowledge base**, and always cites the reference page from the original documents.

---

## 🚀 Features
- **Arabic-first interface** (RTL layout, professional typography).
- Ask natural questions about Smart Mind programs, fees, and policies.
- Strict grounding in the provided knowledge base (**no hallucinations**).
- Every answer includes **page citations** and expandable previews.
- Clean branded UI with Smart Mind logo.

---

## 🗂️ Project Structure

smartmind-info-center/
├─ smartmind_app.py          # Streamlit UI (frontend)
├─ app_core.py               # Core logic: ask(), PAGE_INDEX, verification
├─ kb/
│  └─ v1/
│     └─ knowledge_pack.md   # Knowledge base (prebuilt from DOCX)
├─ logo.png                  # Logo shown in header + sidebar
├─ requirements.txt          # Python dependencies
└─ README.md

---

## ⚙️ Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/smartmind-info-center.git
   cd smartmind-info-center
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run smartmind_app.py

🔑 API Keys

OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
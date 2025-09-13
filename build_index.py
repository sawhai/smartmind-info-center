#!/usr/bin/env python3
"""
Build index during deployment
"""
import os
import sys
from pathlib import Path

def main():
    print("Building search index during deployment...")
    
    # Check for required environment variable
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set up paths
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"
    INDEX_DIR = DATA_DIR / "index"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    DOCX_PATH = PROJECT_ROOT / "docs" / "smartmind_docs_v2.docx"
    
    if not DOCX_PATH.exists():
        print(f"Error: Document not found at {DOCX_PATH}")
        sys.exit(1)
    
    try:
        # Import required libraries
        import json
        import re
        import numpy as np
        from docx import Document
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Helper functions
        def normalize_ar(text: str) -> str:
            if not text:
                return ""
            text = re.sub(r"[\u0640]", "", text)
            text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            return text.strip()

        def read_docx_to_pages(docx_path: Path) -> list:
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

        def chunk_pages(pages: list, target_chars=1800, overlap_chars=250) -> list:
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

        def embed_texts_openai(texts: list) -> np.ndarray:
            EMBED_MODEL = "text-embedding-3-large"
            EMBED_DIM = 3072
            
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

        def save_index(chunks: list, E: np.ndarray):
            META_PATH = INDEX_DIR / "meta.jsonl"
            EMB_PATH = INDEX_DIR / "embeddings.npy"
            
            np.save(EMB_PATH, E)
            with open(META_PATH, "w", encoding="utf-8") as f:
                for ch in chunks:
                    f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        
        # Process the document
        print("Reading DOCX...")
        pages = read_docx_to_pages(DOCX_PATH)
        print(f"Found {len(pages)} pages")
        
        print("Creating chunks...")
        chunks = chunk_pages(pages, target_chars=1800, overlap_chars=250)
        print(f"Created {len(chunks)} chunks")
        
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts_openai(texts)
        print("Embeddings created successfully")
        
        print("Saving index...")
        save_index(chunks, embeddings)
        print("Index built successfully!")
        
    except Exception as e:
        print(f"Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
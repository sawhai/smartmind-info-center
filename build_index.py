#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 17:51:39 2025

@author: ha
"""
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("Building search index during deployment...")
    
    # Import app functions
    from app import read_docx_to_pages, chunk_pages, embed_texts_openai, save_index, PROJECT_ROOT, DOCX_PATH
    
    if not DOCX_PATH.exists():
        print(f"Error: Document not found at {DOCX_PATH}")
        sys.exit(1)
    
    try:
        pages = read_docx_to_pages(DOCX_PATH)
        chunks = chunk_pages(pages, target_chars=1800, overlap_chars=250)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts_openai(texts)
        save_index(chunks, embeddings)
        print("Index built successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
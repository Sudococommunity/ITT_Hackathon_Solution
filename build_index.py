"""
Script to build/rebuild the vector store index.
Run this once before starting the app.

Usage:
    python build_index.py [--force]
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import src.cache_config  # noqa: F401 — forces all caches to D:

from src.slide_parser import parse_pptx
from src.chunker import chunk_projects
from src.embedder import build_vectorstore
from src.config import PPTX_PATH

force = "--force" in sys.argv

print("Step 1: Parsing PPTX slides...")
projects = parse_pptx(PPTX_PATH)
print(f"  Parsed {len(projects)} projects")

print("Step 2: Creating chunks...")
chunks = chunk_projects(projects)
print(f"  Created {len(chunks)} chunks")

print("Step 3: Building vector store (this may take a few minutes on first run)...")
build_vectorstore(chunks, force_rebuild=force)
print("Done!")

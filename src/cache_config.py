"""
Forces ALL model caches (HuggingFace, Transformers, Sentence-Transformers, Torch)
to D: drive under project directory. Import this BEFORE any ML library.
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache", "huggingface")

os.environ["HF_HOME"] = _CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(_CACHE_DIR, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_CACHE_DIR, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(_CACHE_DIR, "sentence_transformers")
os.environ["TORCH_HOME"] = os.path.join(_PROJECT_ROOT, ".cache", "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(_PROJECT_ROOT, ".cache")

# HuggingFace token for authenticated downloads
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Please set it in environment variables.")

os.environ["HF_TOKEN"] = HF_TOKEN

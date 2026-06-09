"""
Configuration for the RAG application.
"""
import os

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama_cloud")  # ollama_local, ollama_cloud, openai_compatible
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Ollama Cloud API
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "9e47782dbbf048d59de1a4702db76881.X03f6HI4OoprXVsTR8lFxEfy")

# For OpenAI-compatible APIs
API_KEY = os.getenv("LLM_API_KEY", "")
API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")

# Retrieval Configuration
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 8

# Data paths
PPTX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset_project_repository.pptx")

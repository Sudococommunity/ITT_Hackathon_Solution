"""
RAG Engine: Retrieval + Re-ranking + LLM Generation.
Heavily grounded — uses hybrid search, metadata pre-filtering,
query analysis, and answer verification to minimize hallucination.
"""
import src.cache_config  # noqa: F401 — must be first to redirect all caches to D:
import re
import json
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedder import query_vectorstore, get_chroma_client, MODEL_NAME


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_all_projects_metadata() -> list[dict]:
    """Load all project metadata from ChromaDB for keyword/metadata search."""
    client = get_chroma_client()
    collection = client.get_collection("projects")
    # Get only full_project chunks (one per project)
    results = collection.get(
        where={"chunk_type": "full_project"},
        include=["documents", "metadatas"],
    )
    projects = []
    for i, doc in enumerate(results['documents']):
        projects.append({
            'id': results['ids'][i],
            'text': doc,
            'metadata': results['metadatas'][i],
        })
    return projects


class RAGEngine:
    def __init__(self, llm_provider="ollama_cloud", llm_model="gemma3:27b",
                 ollama_url="http://localhost:11434", api_key=None, api_base=None):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.api_key = api_key
        self.api_base = api_base

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(MODEL_NAME)

        print("Loading re-ranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Pre-load all project metadata for keyword/filter search
        print("Loading project metadata index...")
        self.all_projects = load_all_projects_metadata()
        print(f"Models loaded. {len(self.all_projects)} projects indexed.")

    # ── QUERY ANALYSIS ──────────────────────────────────────────────────

    def analyze_query(self, query: str) -> dict:
        """
        Analyze query to extract intent, filters, and keywords.
        This is done WITHOUT the LLM — pure rule-based for speed and reliability.
        """
        q = query.lower()
        analysis = {
            'original': query,
            'metadata_filters': {},
            'keywords': [],
            'is_listing_query': False,
            'is_similarity_query': False,
            'is_filter_query': False,
        }

        # Detect listing queries
        if any(w in q for w in ['list', 'show me all', 'show all', 'which projects', 'what projects', 'how many']):
            analysis['is_listing_query'] = True

        # Detect similarity queries
        if any(w in q for w in ['similar to', 'most similar', 'like building', 'resembles']):
            analysis['is_similarity_query'] = True

        # Extract team size filters
        team_match = re.search(r'team\s*(?:size)?\s*(?:was\s+)?(?:under|less than|fewer than|below|<|<=)\s*(\d+)', q)
        if team_match:
            analysis['metadata_filters']['team_size'] = {'$lte': int(team_match.group(1))}
            analysis['is_filter_query'] = True

        team_match2 = re.search(r'team\s*(?:size)?\s*(?:was\s+)?(?:over|more than|greater than|above|>|>=)\s*(\d+)', q)
        if team_match2:
            analysis['metadata_filters']['team_size'] = {'$gte': int(team_match2.group(1))}
            analysis['is_filter_query'] = True

        # Extract duration filters
        dur_match = re.search(r'duration\s*(?:under|less than|below|<|<=)\s*(\d+)\s*months?', q)
        if dur_match:
            analysis['metadata_filters']['duration_months'] = {'$lte': int(dur_match.group(1))}
            analysis['is_filter_query'] = True

        dur_match2 = re.search(r'(?:under|less than|within)\s*(\d+)\s*months?', q)
        if dur_match2 and 'duration_months' not in analysis['metadata_filters']:
            analysis['metadata_filters']['duration_months'] = {'$lte': int(dur_match2.group(1))}
            analysis['is_filter_query'] = True

        # Extract domain keywords
        domain_keywords = [
            'healthcare', 'pharma', 'finance', 'fintech', 'banking', 'insurance',
            'retail', 'e-commerce', 'ecommerce', 'telecom', 'manufacturing',
            'supply chain', 'logistics', 'energy', 'education', 'legal',
            'automotive', 'agriculture', 'media', 'entertainment', 'government',
            'real estate', 'construction', 'aerospace', 'defense', 'life sciences',
        ]
        for kw in domain_keywords:
            if kw in q:
                analysis['keywords'].append(kw)

        # Extract technology keywords
        tech_keywords = [
            'nlp', 'natural language', 'llm', 'large language model', 'gpt',
            'bert', 'transformer', 'genai', 'generative ai', 'computer vision',
            'deep learning', 'machine learning', 'ml', 'ai', 'data engineering',
            'etl', 'data pipeline', 'iot', 'blockchain', 'cloud', 'aws', 'azure',
            'gcp', 'python', 'spark', 'kafka', 'docker', 'kubernetes', 'react',
            'tensorflow', 'pytorch', 'langchain', 'rag', 'vector', 'embedding',
            'recommendation', 'churn', 'fraud', 'anomaly', 'forecasting',
            'time series', 'graph', 'neo4j', 'knowledge graph', 'chatbot',
            'sentiment', 'ocr', 'image', 'speech', 'audio',
        ]
        for kw in tech_keywords:
            if kw in q:
                analysis['keywords'].append(kw)

        return analysis

    # ── KEYWORD/METADATA SEARCH (Hybrid) ────────────────────────────────

    def keyword_search(self, query_analysis: dict) -> list[dict]:
        """
        Search projects using exact keyword matching on text and metadata.
        This complements vector search for precise filtering queries.
        """
        results = []
        keywords = query_analysis['keywords']
        filters = query_analysis['metadata_filters']
        q_lower = query_analysis['original'].lower()

        for proj in self.all_projects:
            text_lower = proj['text'].lower()
            meta = proj['metadata']

            # Check metadata filters
            passes_filters = True
            for field, condition in filters.items():
                val = meta.get(field, 0)
                if isinstance(val, str):
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        val = 0
                if isinstance(condition, dict):
                    for op, threshold in condition.items():
                        if op == '$lte' and val > threshold:
                            passes_filters = False
                        elif op == '$gte' and val < threshold:
                            passes_filters = False

            if not passes_filters:
                continue

            # Score by keyword matches
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    score += 2
                if kw in meta.get('domain', '').lower():
                    score += 3  # Domain match is very strong
                if kw in meta.get('technologies', '').lower():
                    score += 2
                if kw in meta.get('project_type', '').lower():
                    score += 2

            # If we have filters but no keywords, include all passing projects
            if filters and not keywords:
                score = 1

            if score > 0:
                results.append({
                    'text': proj['text'],
                    'metadata': proj['metadata'],
                    'keyword_score': score,
                })

        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results

    # ── VECTOR SEARCH ───────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20, where_filter: dict = None) -> list[dict]:
        """Retrieve relevant chunks from vector store."""
        results = query_vectorstore(
            query=query,
            model=self.embed_model,
            n_results=top_k,
            where_filter=where_filter,
        )

        chunks = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                chunks.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                })
        return chunks

    # ── HYBRID MERGE ────────────────────────────────────────────────────

    def hybrid_merge(self, vector_results: list[dict], keyword_results: list[dict],
                     vector_weight: float = 0.6) -> list[dict]:
        """
        Merge vector search and keyword search results using reciprocal rank fusion.
        """
        scores = {}  # project_id -> {'score': float, 'chunk': dict}
        k = 60  # RRF constant

        # Score vector results
        for rank, chunk in enumerate(vector_results):
            pid = chunk['metadata'].get('project_id', chunk.get('id', str(rank)))
            rrf = vector_weight / (k + rank + 1)
            if pid not in scores or rrf > scores[pid]['score']:
                scores[pid] = {'score': scores.get(pid, {}).get('score', 0) + rrf, 'chunk': chunk}
            else:
                scores[pid]['score'] += rrf

        # Score keyword results
        kw_weight = 1.0 - vector_weight
        for rank, chunk in enumerate(keyword_results):
            pid = chunk['metadata'].get('project_id', str(rank))
            rrf = kw_weight / (k + rank + 1)
            if pid not in scores:
                scores[pid] = {'score': rrf, 'chunk': chunk}
            else:
                scores[pid]['score'] += rrf

        # Sort by combined score
        merged = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in merged]

    # ── RE-RANKING ──────────────────────────────────────────────────────

    def rerank(self, query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
        """Re-rank retrieved chunks using cross-encoder."""
        if not chunks:
            return []

        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])

        chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        return chunks[:top_k]

    def deduplicate_by_project(self, chunks: list[dict]) -> list[dict]:
        """Keep best chunk per project to avoid redundancy in context."""
        seen_projects = {}
        deduped = []
        for chunk in chunks:
            pid = chunk['metadata'].get('project_id', '')
            if pid not in seen_projects:
                seen_projects[pid] = True
                deduped.append(chunk)
            elif chunk['metadata'].get('chunk_type') == 'full_project' and pid in seen_projects:
                for j, d in enumerate(deduped):
                    if d['metadata'].get('project_id') == pid and d['metadata'].get('chunk_type') != 'full_project':
                        deduped[j] = chunk
                        break
        return deduped

    # ── CONTEXT & PROMPT ────────────────────────────────────────────────

    def build_context(self, chunks: list[dict]) -> str:
        """Build structured context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            meta = chunk['metadata']
            slides = meta.get('slide_numbers', 'N/A')
            context_parts.append(
                f"[Source #{i+1}: Project {meta.get('project_id', 'N/A')} - "
                f"{meta.get('title', 'N/A')} | Domain: {meta.get('domain', 'N/A')} | "
                f"Slides: {slides} | Team: {meta.get('team_size', 'N/A')} | "
                f"Duration: {meta.get('duration_months', 'N/A')}m]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, query: str, context: str, query_analysis: dict) -> str:
        """Build grounded LLM prompt with strict anti-hallucination instructions."""

        # Add specific instructions based on query type
        type_instructions = ""
        if query_analysis['is_listing_query']:
            type_instructions = """
LISTING QUERY DETECTED: The user wants a list. Present ALL matching projects from the context as a numbered/bulleted list. Include project ID, title, domain, key details, and slide numbers for each. Do NOT omit any matching project from the context."""

        if query_analysis['is_filter_query']:
            filter_desc = []
            for field, cond in query_analysis['metadata_filters'].items():
                for op, val in cond.items():
                    op_word = "at most" if op == "$lte" else "at least"
                    filter_desc.append(f"{field.replace('_', ' ')}: {op_word} {val}")
            type_instructions += f"\nFILTER QUERY DETECTED: Results are already pre-filtered by: {', '.join(filter_desc)}. All projects shown in context pass these filters."

        if query_analysis['is_similarity_query']:
            type_instructions += """
SIMILARITY QUERY DETECTED: Explain specifically what makes each project similar — compare problem domain, technologies, approach, and outcomes."""

        return f"""You are a Project Intelligence Assistant. Your ONLY job is to answer based on the CONTEXT below.

STRICT RULES:
1. ONLY use information explicitly stated in the CONTEXT below. If it's not in the context, DO NOT say it.
2. Every fact you state MUST have a source citation: [Project ID, Slides X,Y].
3. If the context doesn't contain enough info, say: "Based on the retrieved projects, I don't have sufficient information to answer this fully."
4. NEVER invent project names, IDs, technologies, numbers, or outcomes.
5. When listing projects, use this format for each:
   - **[Project ID] Project Title** (Slides: X,Y)
     Domain: ... | Team: ... | Duration: ...
     Key details...
6. If a field value is not in the context, write "Not specified" — do NOT guess.
{type_instructions}

CONTEXT (Retrieved Project Data — this is your ONLY source of truth):
{context}

USER QUESTION: {query}

GROUNDED ANSWER (cite every claim with [Project ID, Slides]):"""

    # ── LLM CALLS ───────────────────────────────────────────────────────

    def call_llm(self, prompt: str) -> str:
        if self.llm_provider == "ollama_local":
            return self._call_ollama_local(prompt)
        elif self.llm_provider == "ollama_cloud":
            return self._call_ollama_cloud(prompt)
        elif self.llm_provider == "openai_compatible":
            return self._call_openai_compatible(prompt)
        else:
            return f"Unknown LLM provider: {self.llm_provider}"

    def _call_ollama_local(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("response", "No response from model.")
        except requests.ConnectionError:
            return "ERROR: Cannot connect to Ollama. Make sure Ollama is running."
        except Exception as e:
            return f"ERROR calling Ollama: {str(e)}"

    def _call_ollama_cloud(self, prompt: str) -> str:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            response = requests.post(
                "https://api.ollama.com/api/chat",
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a factual assistant. ONLY answer from provided context. Never add information not in the context. Always cite sources."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            return content or "No response from model."
        except requests.ConnectionError:
            return "ERROR: Cannot connect to Ollama Cloud API."
        except requests.Timeout:
            return "ERROR: Ollama Cloud API timed out. Try a smaller model or try again."
        except Exception as e:
            resp_text = ""
            if hasattr(e, 'response') and e.response is not None:
                resp_text = e.response.text
            return f"ERROR calling Ollama Cloud: {str(e)} {resp_text}"

    def _call_openai_compatible(self, prompt: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            base = self.api_base or "http://localhost:8000/v1"
            response = requests.post(
                f"{base}/chat/completions",
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a factual assistant. ONLY answer from provided context. Never add information not in the context."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.ConnectionError:
            return f"ERROR: Cannot connect to API at {base}."
        except Exception as e:
            return f"ERROR calling API: {str(e)}"

    # ── ANSWER VERIFICATION ─────────────────────────────────────────────

    def verify_answer(self, answer: str, chunks: list[dict]) -> dict:
        """
        Verify that the answer is grounded in the retrieved context.
        Checks if project IDs mentioned in the answer exist in the context.
        Returns verification report.
        """
        # Extract project IDs mentioned in the answer
        answer_pids = set(re.findall(r'P\d{3}', answer))

        # Project IDs in context
        context_pids = set()
        for chunk in chunks:
            pid = chunk['metadata'].get('project_id', '')
            if pid:
                context_pids.add(pid)

        # Check for hallucinated project references
        hallucinated_pids = answer_pids - context_pids
        grounded_pids = answer_pids & context_pids

        grounding_ratio = len(grounded_pids) / len(answer_pids) if answer_pids else 1.0

        return {
            'grounding_ratio': grounding_ratio,
            'grounded_projects': sorted(grounded_pids),
            'hallucinated_projects': sorted(hallucinated_pids),
            'is_grounded': len(hallucinated_pids) == 0,
        }

    # ── MAIN QUERY PIPELINE ─────────────────────────────────────────────

    def query(self, user_query: str, top_k_retrieve: int = 20, top_k_rerank: int = 10,
              where_filter: dict = None) -> dict:
        """
        Full RAG pipeline:
        1. Analyze query (extract filters, keywords, intent)
        2. Hybrid search (vector + keyword)
        3. Re-rank with cross-encoder
        4. Deduplicate by project
        5. Generate grounded answer
        6. Verify answer grounding
        """
        # Step 1: Analyze query
        query_analysis = self.analyze_query(user_query)

        # Step 2a: Vector search
        # Auto-apply metadata filters extracted from query
        combined_filter = where_filter
        auto_filters = query_analysis['metadata_filters']
        if auto_filters:
            filter_list = []
            if combined_filter:
                filter_list.append(combined_filter)
            for field, condition in auto_filters.items():
                filter_list.append({field: condition})
            if len(filter_list) == 1:
                combined_filter = filter_list[0]
            elif len(filter_list) > 1:
                combined_filter = {"$and": filter_list}

        vector_chunks = self.retrieve(user_query, top_k=top_k_retrieve, where_filter=combined_filter)

        # If filter gave too few results, retry without metadata filter
        if len(vector_chunks) < 3 and combined_filter:
            vector_chunks = self.retrieve(user_query, top_k=top_k_retrieve)

        # Step 2b: Keyword search
        keyword_chunks = self.keyword_search(query_analysis)

        # Step 3: Hybrid merge
        if keyword_chunks:
            # For filter-heavy queries, weight keyword search higher
            kw_weight = 0.7 if query_analysis['is_filter_query'] else 0.4
            merged = self.hybrid_merge(vector_chunks, keyword_chunks, vector_weight=1.0 - kw_weight)
        else:
            merged = vector_chunks

        if not merged:
            return {
                'answer': "No relevant projects found for your query.",
                'sources': [],
                'confidence': 0.0,
                'verification': {'is_grounded': True, 'grounding_ratio': 1.0},
            }

        # Step 4: Re-rank
        reranked = self.rerank(user_query, merged, top_k=top_k_rerank)

        # Step 5: Deduplicate by project
        deduped = self.deduplicate_by_project(reranked)

        # Step 6: Build context and generate
        context = self.build_context(deduped)
        prompt = self.build_prompt(user_query, context, query_analysis)
        answer = self.call_llm(prompt)

        # Step 7: Verify answer grounding
        verification = self.verify_answer(answer, deduped)

        # If hallucinated projects found, add warning
        if not verification['is_grounded']:
            hallucinated = ', '.join(verification['hallucinated_projects'])
            answer += f"\n\n⚠️ **Warning**: The following project IDs were mentioned but NOT found in retrieved context: {hallucinated}. These references may be inaccurate."

        # Extract sources
        sources = []
        seen = set()
        for chunk in deduped:
            pid = chunk['metadata'].get('project_id', '')
            if pid not in seen:
                seen.add(pid)
                sources.append({
                    'project_id': pid,
                    'title': chunk['metadata'].get('title', ''),
                    'domain': chunk['metadata'].get('domain', ''),
                    'slides': chunk['metadata'].get('slide_numbers', ''),
                    'relevance_score': round(chunk.get('rerank_score', 0), 4),
                })

        # Confidence score
        if reranked:
            top_scores = [c.get('rerank_score', 0) for c in reranked[:3]]
            avg_score = sum(top_scores) / len(top_scores)
            confidence = min(max((avg_score + 5) / 10, 0), 1.0)
        else:
            confidence = 0.0

        return {
            'answer': answer,
            'sources': sources,
            'confidence': round(confidence, 3),
            'num_chunks_retrieved': len(vector_chunks),
            'num_keyword_matches': len(keyword_chunks),
            'num_chunks_after_rerank': len(reranked),
            'num_projects_in_context': len(deduped),
            'query_analysis': query_analysis,
            'verification': verification,
        }

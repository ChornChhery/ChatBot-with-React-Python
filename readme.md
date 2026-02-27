# ChatBotRagPy 🤖

A Retrieval-Augmented Generation (RAG) Chatbot built with **FastAPI + React**.
Upload documents and chat with them in real time using local Ollama models.

> Same features as ChatBotRAG (.NET version) — rebuilt in Python + React.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite (port 5173) |
| Backend | FastAPI Python (port 8000) |
| Real-time | WebSocket |
| Database | SQL Server LocalDB |
| ORM | SQLAlchemy + Alembic |
| AI Chat | Ollama — llama3.2:3b |
| AI Embedding | Ollama — mxbai-embed-large |

---

## ⚡ Already Cloned? Start Here

> If you just cloned this repo, follow **only these steps**. Do NOT follow the "Build From Scratch" section below — those files already exist in the repo.

### 1. Install prerequisites (one time only)

- [Python 3.12+](https://www.python.org/downloads/)
- [Node.js 24+](https://nodejs.org/)
- [Ollama](https://ollama.com/download)
- [SQL Server LocalDB](https://aka.ms/sqllocaldb)
- [ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

### 2. Pull Ollama models (one time only)

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### 3. Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Create the .env file

> ⚠️ This file is NOT in the repo for security reasons. You must create it manually.

Create a new file at `backend/.env` and paste this content:

```dotenv
DATABASE_URL=mssql+pyodbc://@(localdb)\MSSQLLocalDB/ChatBotRagPy?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=mxbai-embed-large:latest
VECTOR_WEIGHT=0.7
MIN_SIMILARITY_THRESHOLD=0.60
```

> ⚠️ Always use a **single backslash** `\` — never `\\`.

### 5. Create the database

```bash
sqlcmd -S "(localdb)\MSSQLLocalDB" -E -Q "CREATE DATABASE ChatBotRagPy;"
```

### 6. Run database migrations

```bash
alembic upgrade head
```

You should see:
```
Running upgrade  -> xxxxxxxx, InitialCreate
```
No error = ✅ success

### 7. Set up the frontend

```bash
cd ..\frontend
npm install
```

### 8. Run the project

Open **3 separate terminals**:

**Terminal 1 — Ollama:**
```bash
ollama serve
```

**Terminal 2 — Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

**Terminal 3 — Frontend:**
```bash
cd frontend
npm run dev
```

Open browser: **http://localhost:5173** ✅

---

## 🏗️ Build From Scratch

> Only follow this section if you are setting up this project **for the first time** on a new machine without cloning.

### Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [Node.js 24+](https://nodejs.org/)
- [Ollama](https://ollama.com/download)
- [SQL Server LocalDB](https://aka.ms/sqllocaldb) (comes with Visual Studio)
- [ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

---

### Step 1 — Pull Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

Wait for both to finish before continuing.

---

### Step 2 — Create Project Folders

```bash
cd D:\Jame
mkdir ChatbotRagPy
cd ChatbotRagPy
mkdir backend frontend
```

---

### Step 3 — Backend Setup

#### 3.1 Create Virtual Environment

```bash
cd D:\Jame\ChatbotRagPy\backend
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal line.

#### 3.2 Install Python Dependencies

```bash
pip install fastapi "uvicorn[standard]" sqlalchemy alembic pyodbc python-multipart aiofiles ollama httpx numpy rank-bm25 pypdf2 markdownify python-dotenv "pydantic-settings"
pip freeze > requirements.txt
```

#### 3.3 Create All Folders

```bash
mkdir app
mkdir app\core
mkdir app\models
mkdir app\schemas
mkdir app\routers
mkdir app\services
mkdir app\chunking
mkdir app\evaluators
```

#### 3.4 Create All Empty __init__.py Files

```bash
type nul > app\__init__.py
type nul > app\core\__init__.py
type nul > app\models\__init__.py
type nul > app\schemas\__init__.py
type nul > app\routers\__init__.py
type nul > app\services\__init__.py
type nul > app\chunking\__init__.py
type nul > app\evaluators\__init__.py
```

#### 3.5 Create .env File

Create `backend/.env` with this content:

```dotenv
DATABASE_URL=mssql+pyodbc://@(localdb)\MSSQLLocalDB/ChatBotRagPy?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=mxbai-embed-large:latest
VECTOR_WEIGHT=0.7
MIN_SIMILARITY_THRESHOLD=0.60
```

> ⚠️ Use a **single backslash** `\` — never `\\` in the .env file.

#### 3.6 Create app/core/config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embed_model: str = "mxbai-embed-large:latest"
    vector_weight: float = 0.7
    min_similarity_threshold: float = 0.60

    class Config:
        env_file = ".env"

settings = Settings()
```

#### 3.7 Create app/core/enums.py

```python
from enum import IntEnum

class ChunkingStrategy(IntEnum):
    FIXED_SIZE = 0
    CONTENT_AWARE = 1
    SEMANTIC = 2

class DocumentStatus:
    UPLOADING = "Uploading"
    PROCESSING = "Processing"
    READY = "Ready"
    FAILED = "Failed"
```

> ⚠️ Do NOT write `from enum import IntEnum, str as StrEnum` — that causes an ImportError in Python 3.12.

#### 3.8 Create app/database.py

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### 3.9 Create app/models/document.py

```python
import uuid
import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_name = Column(String(500), nullable=False)
    status = Column(String(50), default="Uploading")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
```

#### 3.10 Create app/models/document_chunk.py

```python
import uuid
from sqlalchemy import Column, String, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunking_method = Column(String(50), default="FixedSize")
    embedding_json = Column(Text, nullable=True)

    document = relationship("Document", back_populates="chunks")
```

#### 3.11 Create app/schemas/document.py

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentDto(BaseModel):
    id: str
    file_name: str
    status: str
    created_at: datetime
    chunk_count: Optional[int] = 0

    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    id: str
    file_name: str
    message: str

class CacheStats(BaseModel):
    totalChunks: int
    totalDocuments: int
    isLoaded: bool
    estimatedMemoryMb: float
```

#### 3.12 Create app/schemas/chat.py

```python
from pydantic import BaseModel
from typing import List, Optional

class ChatMessageDto(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessageDto]] = []

class DocumentChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    content: str
    score: float
```

#### 3.13 Create app/schemas/evaluation.py

```python
from pydantic import BaseModel
from typing import Optional, List

class EvaluationRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 5

class EvaluationResult(BaseModel):
    question: str
    auto_generated_reference: str
    generated_answer: str
    source_documents: List[str]
    bleu_score: float
    gleu_score: float
    f1_score: float
    llm_judge_score: float
    llm_judge_explanation: str
    overall_score: float
```

#### 3.14 Create app/chunking/base.py

```python
from abc import ABC, abstractmethod
from typing import List

class BaseChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass
```

#### 3.15 Create app/chunking/fixed_size.py

```python
from typing import List
from app.chunking.base import BaseChunkingStrategy

class FixedSizeChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.overlap
        return [c for c in chunks if c]
```

#### 3.16 Create app/chunking/content_aware.py

```python
from typing import List
from app.chunking.base import BaseChunkingStrategy
import re

class ContentAwareChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 100, max_size: int = 1000):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n#{1,6} |\n\n|(?<=។)', text)
        chunks, current = [], ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.max_size:
                current += " " + para
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = para
        if current.strip():
            chunks.append(current.strip())
        return chunks
```

#### 3.17 Create app/chunking/semantic.py

```python
from typing import List
from app.chunking.base import BaseChunkingStrategy
import re

class SemanticChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 150, max_size: int = 1200, threshold: float = 0.3):
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+|(?<=។)\s*|(?<=[ๆฯ])\s+|\n', text)

    def _overlap(self, a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / min(len(sa), len(sb))

    def chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._split_sentences(text) if s.strip()]
        if not sentences:
            return []
        chunks, current = [], sentences[0]
        for sent in sentences[1:]:
            if (len(current) + len(sent) <= self.max_size and
                    self._overlap(current, sent) >= self.threshold):
                current += " " + sent
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks
```

#### 3.18 Create app/evaluators/bleu.py

```python
from collections import Counter
import math

class BLEUEvaluator:
    def score(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            match = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(match / total if total > 0 else 0.0)
        if all(p == 0 for p in precisions):
            return 0.0
        log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if hyp_tokens else 0.0
        return round(bp * math.exp(log_avg), 4)
```

#### 3.19 Create app/evaluators/gleu.py

```python
from collections import Counter

class GLEUEvaluator:
    def score(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        total_match = total_ref = total_hyp = 0
        for n in range(1, max_n + 1):
            ref_ng = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            hyp_ng = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            total_match += sum((ref_ng & hyp_ng).values())
            total_ref += sum(ref_ng.values())
            total_hyp += sum(hyp_ng.values())
        if total_ref == 0 or total_hyp == 0:
            return 0.0
        precision = total_match / total_hyp
        recall = total_match / total_ref
        return round((precision + recall) / 2, 4)
```

#### 3.20 Create app/evaluators/f1.py

```python
class F1Evaluator:
    def score(self, reference: str, hypothesis: str) -> float:
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not ref_tokens or not hyp_tokens:
            return 0.0
        common = ref_tokens & hyp_tokens
        if not common:
            return 0.0
        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)
        return round(2 * precision * recall / (precision + recall), 4)
```

#### 3.21 Create app/services/embedding_cache.py

```python
import json
import numpy as np
from threading import Lock
from typing import Dict, List

class EmbeddingCacheService:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._lock = Lock()
        self._loaded = False

    def warm_up(self, db):
        from app.models.document_chunk import DocumentChunk
        from app.models.document import Document
        chunks = db.query(DocumentChunk).join(Document)\
            .filter(Document.status == "Ready").all()
        with self._lock:
            for chunk in chunks:
                if chunk.embedding_json:
                    self._cache[chunk.id] = {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "vector": np.array(json.loads(chunk.embedding_json), dtype=np.float32)
                    }
            self._loaded = True

    def get_all(self) -> List[dict]:
        return list(self._cache.values())

    def add_chunks(self, chunks: List[dict]):
        with self._lock:
            for chunk in chunks:
                self._cache[chunk["id"]] = chunk

    def remove_document(self, document_id: str):
        with self._lock:
            keys = [k for k, v in self._cache.items() if v["document_id"] == document_id]
            for k in keys:
                del self._cache[k]

    @property
    def stats(self):
        total = len(self._cache)
        docs = len(set(v["document_id"] for v in self._cache.values())) if self._cache else 0
        mem_mb = round(total * 4 * 1024 / (1024**2), 2)
        return {
            "totalChunks": total,
            "totalDocuments": docs,
            "isLoaded": self._loaded,
            "estimatedMemoryMb": mem_mb
        }

embedding_cache = EmbeddingCacheService()
```

#### 3.22 Create app/services/embedding_service.py

```python
import json
import httpx
from typing import List
from app.core.config import settings

class EmbeddingService:
    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60) as client:
            response = await client.post("/api/embeddings", json={
                "model": settings.embed_model,
                "prompt": text
            })
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results

embedding_service = EmbeddingService()
```

#### 3.23 Create app/services/bm25_service.py

```python
import re
from typing import List

class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _detect_language(self, text: str) -> str:
        if re.search(r'[\u0E00-\u0E7F]', text):
            return 'thai'
        if re.search(r'[\u1780-\u17FF]', text):
            return 'khmer'
        return 'english'

    def _tokenize(self, text: str) -> List[str]:
        lang = self._detect_language(text)
        if lang in ('thai', 'khmer'):
            n = 3
            return [text[i:i+n] for i in range(len(text)-n+1)]
        stop_words = {'the','a','an','is','in','on','at','to','for','of','and','or'}
        return [w for w in re.findall(r'\w+', text.lower()) if w not in stop_words]

    def score(self, query: str, documents: List[str]) -> List[float]:
        import math
        tokenized_docs = [self._tokenize(d) for d in documents]
        avg_len = sum(len(d) for d in tokenized_docs) / len(tokenized_docs) if tokenized_docs else 1
        query_tokens = self._tokenize(query)
        N = len(documents)
        scores = []
        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            score = 0.0
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            for qt in query_tokens:
                tf = tf_map.get(qt, 0)
                df = sum(1 for d in tokenized_docs if qt in d)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
            scores.append(score)
        max_score = max(scores) if scores else 1
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

bm25_service = BM25Service()
```

#### 3.24 Create app/services/hybrid_search.py

```python
import numpy as np
from typing import List, Dict
from app.services.embedding_cache import embedding_cache
from app.services.bm25_service import bm25_service
from app.core.config import settings

class HybridSearchService:
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def search(self, query_vector: List[float], query_text: str, top_k: int = 5, document_id: str = None) -> List[Dict]:
        cached = embedding_cache.get_all()
        if document_id:
            cached = [c for c in cached if c["document_id"] == document_id]
        if not cached:
            return []
        qv = np.array(query_vector, dtype=np.float32)
        contents = [c["content"] for c in cached]
        bm25_scores = bm25_service.score(query_text, contents)
        vw = settings.vector_weight
        results = []
        for i, chunk in enumerate(cached):
            vec_score = self.cosine_similarity(qv, chunk["vector"])
            final_score = vw * vec_score + (1 - vw) * bm25_scores[i]
            if final_score >= settings.min_similarity_threshold:
                results.append({**chunk, "score": final_score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

hybrid_search_service = HybridSearchService()
```

#### 3.25 Create app/services/document_service.py

```python
import json
import uuid
from typing import List
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.core.enums import DocumentStatus, ChunkingStrategy
from app.services.embedding_service import embedding_service
from app.services.embedding_cache import embedding_cache
from app.chunking.fixed_size import FixedSizeChunkingStrategy
from app.chunking.content_aware import ContentAwareChunkingStrategy
from app.chunking.semantic import SemanticChunkingStrategy
import numpy as np

STRATEGY_MAP = {
    ChunkingStrategy.FIXED_SIZE: FixedSizeChunkingStrategy,
    ChunkingStrategy.CONTENT_AWARE: ContentAwareChunkingStrategy,
    ChunkingStrategy.SEMANTIC: SemanticChunkingStrategy,
}

STRATEGY_NAME_MAP = {
    ChunkingStrategy.FIXED_SIZE: "FixedSize",
    ChunkingStrategy.CONTENT_AWARE: "ContentAware",
    ChunkingStrategy.SEMANTIC: "Semantic",
}

class DocumentService:
    async def process_document(self, db: Session, document_id: str, text: str, strategy: ChunkingStrategy):
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return
        try:
            doc.status = DocumentStatus.PROCESSING
            db.commit()
            chunker = STRATEGY_MAP[strategy]()
            chunks = chunker.chunk(text)
            method_name = STRATEGY_NAME_MAP[strategy]
            new_cache_chunks = []
            for i, chunk_text in enumerate(chunks):
                vector = await embedding_service.embed(chunk_text)
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=i,
                    chunking_method=method_name,
                    embedding_json=json.dumps(vector)
                )
                db.add(chunk)
                new_cache_chunks.append({
                    "id": chunk.id,
                    "document_id": document_id,
                    "content": chunk_text,
                    "vector": np.array(vector, dtype=np.float32)
                })
            doc.status = DocumentStatus.READY
            db.commit()
            embedding_cache.add_chunks(new_cache_chunks)
        except Exception as e:
            doc.status = DocumentStatus.FAILED
            db.commit()
            print(f"[DocumentService] Error: {e}")

    def delete_document(self, db: Session, document_id: str):
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            embedding_cache.remove_document(document_id)
```

#### 3.26 Create app/services/rag_service.py

```python
from typing import List, AsyncGenerator
import httpx
from app.services.embedding_service import embedding_service
from app.services.hybrid_search import hybrid_search_service
from app.core.config import settings

class RagService:
    def __init__(self):
        self._last_sources = []

    async def stream_answer(self, question: str, history: list = [], document_id: str = None) -> AsyncGenerator[str, None]:
        query_vector = await embedding_service.embed(question)
        chunks = await hybrid_search_service.search(query_vector, question, top_k=5, document_id=document_id)
        self._last_sources = chunks
        context = "\n\n".join([f"[Source {i+1}]: {c['content']}" for i, c in enumerate(chunks)])
        messages = [{"role": "system", "content": f"You are a helpful assistant. Use the following context to answer:\n\n{context}"}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as client:
            async with client.stream("POST", "/api/chat", json={
                "model": settings.chat_model,
                "messages": messages,
                "stream": True
            }) as response:
                import json
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if token := data.get("message", {}).get("content", ""):
                            yield token
                        if data.get("done"):
                            break

    def get_last_sources(self):
        return [
            {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "content": c["content"][:200],
                "score": round(c["score"], 4)
            }
            for c in self._last_sources
        ]
```

#### 3.27 Create app/services/evaluation_service.py

```python
import httpx
import json
from app.services.embedding_service import embedding_service
from app.services.hybrid_search import hybrid_search_service
from app.evaluators.bleu import BLEUEvaluator
from app.evaluators.gleu import GLEUEvaluator
from app.evaluators.f1 import F1Evaluator
from app.core.config import settings

bleu = BLEUEvaluator()
gleu = GLEUEvaluator()
f1 = F1Evaluator()

class EvaluationService:
    async def _generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as client:
            response = await client.post("/api/chat", json={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })
            return response.json()["message"]["content"]

    async def evaluate(self, question: str, document_id: str = None, top_k: int = 5):
        query_vector = await embedding_service.embed(question)
        chunks = await hybrid_search_service.search(query_vector, question, top_k=top_k, document_id=document_id)
        context = "\n\n".join([c["content"] for c in chunks])
        source_docs = list(set(c["document_id"] for c in chunks))
        reference = await self._generate(f"Based on this context, write an ideal answer to: {question}\n\nContext:\n{context}")
        rag_answer = await self._generate(f"Answer this question using only the context provided.\n\nQuestion: {question}\n\nContext:\n{context}")
        judge_prompt = f"Score this answer from 0-10.\n\nQuestion: {question}\nReference: {reference}\nAnswer: {rag_answer}\n\nRespond in JSON: {{\"score\": 8, \"explanation\": \"...\"}}"
        judge_raw = await self._generate(judge_prompt)
        try:
            judge_data = json.loads(judge_raw)
            judge_score = judge_data.get("score", 5) / 10
            judge_explanation = judge_data.get("explanation", "")
        except:
            judge_score = 0.5
            judge_explanation = judge_raw
        bleu_score = bleu.score(reference, rag_answer)
        gleu_score = gleu.score(reference, rag_answer)
        f1_score = f1.score(reference, rag_answer)
        overall = round((bleu_score + gleu_score + f1_score + judge_score) / 4, 4)
        return {
            "question": question,
            "auto_generated_reference": reference,
            "generated_answer": rag_answer,
            "source_documents": source_docs,
            "bleu_score": bleu_score,
            "gleu_score": gleu_score,
            "f1_score": f1_score,
            "llm_judge_score": judge_score,
            "llm_judge_explanation": judge_explanation,
            "overall_score": overall
        }
```

#### 3.28 Create app/routers/documents.py

```python
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.document_service import DocumentService
from app.services.embedding_cache import embedding_cache
from app.core.enums import ChunkingStrategy
import PyPDF2
import io

router = APIRouter()
document_service = DocumentService()

def extract_text(filename: str, content: bytes) -> str:
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")

@router.get("")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    result = []
    for doc in docs:
        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        result.append({
            "id": doc.id, "file_name": doc.file_name,
            "status": doc.status, "created_at": doc.created_at,
            "chunk_count": chunk_count
        })
    return result

@router.get("/cache-stats")
def cache_stats():
    return embedding_cache.stats

@router.get("/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: int = Query(default=0),
    db: Session = Depends(get_db)
):
    content = await file.read()
    text = extract_text(file.filename, content)
    doc = Document(id=str(uuid.uuid4()), file_name=file.filename)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    chunk_strategy = ChunkingStrategy(strategy)
    background_tasks.add_task(document_service.process_document, db, doc.id, text, chunk_strategy)
    return {"id": doc.id, "file_name": doc.file_name, "message": "Upload started"}

@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document_service.delete_document(db, document_id)
```

#### 3.29 Create app/routers/chat.py

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.rag_service import RagService
import json

router = APIRouter()

@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    rag = RagService()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            async for token in rag.stream_answer(
                request["question"],
                request.get("history", []),
                request.get("document_id")
            ):
                await websocket.send_text(json.dumps({"token": token, "isFinal": False}))
            sources = rag.get_last_sources()
            await websocket.send_text(json.dumps({"token": "", "isFinal": True, "sources": sources}))
    except WebSocketDisconnect:
        pass
```

#### 3.30 Create app/routers/evaluation.py

```python
from fastapi import APIRouter
from app.schemas.evaluation import EvaluationRequest
from app.services.evaluation_service import EvaluationService

router = APIRouter()
evaluation_service = EvaluationService()

@router.post("")
async def run_evaluation(request: EvaluationRequest):
    return await evaluation_service.evaluate(
        request.question,
        request.document_id,
        request.top_k
    )
```

#### 3.31 Create app/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.database import Base
from app.routers import documents, chat, evaluation
from app.services.embedding_cache import embedding_cache

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ChatBot RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])

@app.on_event("startup")
async def startup():
    db = SessionLocal()
    try:
        embedding_cache.warm_up(db)
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "ChatBot RAG API is running"}
```

#### 3.32 Initialize Alembic

```bash
alembic init alembic
```

Then open `alembic/env.py` and add these two lines after the existing imports at the top:

```python
from app.core.config import settings
config.set_main_option("sqlalchemy.url", settings.database_url)
```

Also make sure this is in `alembic/env.py` to detect your models:

```python
from app.database import Base
from app.models import document, document_chunk
target_metadata = Base.metadata
```

#### 3.33 Create the Database

```bash
sqlcmd -S "(localdb)\MSSQLLocalDB" -E -Q "CREATE DATABASE ChatBotRagPy;"
```

#### 3.34 Run Migrations

```bash
alembic revision --autogenerate -m "InitialCreate"
alembic upgrade head
```

You should see:
```
Running upgrade  -> xxxxxxxx, InitialCreate
```
No error = ✅ success

---

### Step 4 — Frontend Setup

#### 4.1 Create React App

```bash
cd D:\Jame\ChatbotRagPy\frontend
npm create vite@latest . -- --template react
```

When asked **"Use Vite 8 beta?"** → select **No**
When asked **"Install with npm and start now?"** → select **Yes**

#### 4.2 Install Dependencies

```bash
npm install
npm install axios react-router-dom
```

#### 4.3 Create Folders

```bash
mkdir src\pages
mkdir src\components
```

#### 4.4 Create Frontend Files

Create these files in `frontend/src/`:

| File | Purpose |
|---|---|
| `main.jsx` | React entry point |
| `App.jsx` | Router + layout |
| `App.css` | Layout styles |
| `index.css` | Global dark theme |
| `components/Sidebar.jsx` | Left navigation |
| `components/Sidebar.css` | Sidebar styles |
| `pages/Chat.jsx` | Chat page |
| `pages/Chat.css` | Chat styles |
| `pages/Rag.jsx` | Documents page |
| `pages/Rag.css` | Documents styles |
| `pages/Evaluation.jsx` | Evaluation page |
| `pages/Evaluation.css` | Evaluation styles |

---

### Step 5 — Open in VS Code

```bash
cd D:\Jame\ChatbotRagPy
code .
```

---

## Running the Project

Open **3 separate terminals** every time you want to run the project:

### Terminal 1 — Ollama

```bash
ollama serve
```

### Terminal 2 — Backend

```bash
cd D:\Jame\ChatbotRagPy\backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

Wait for: `Uvicorn running on http://127.0.0.1:8000`

### Terminal 3 — Frontend

```bash
cd D:\Jame\ChatbotRagPy\frontend
npm run dev
```

Wait for: `VITE ready — http://localhost:5173/`

### Open Browser

```
http://localhost:5173
```

---

## Port Reference

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| WebSocket | ws://localhost:8000/api/chat/ws |
| Ollama | http://localhost:11434 |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `cannot import name 'str' from enum` | Wrong import in enums.py | Use only `from enum import IntEnum` |
| `No module named app.chunking.fixed_size` | File missing | Create `fixed_size.py` in `app/chunking/` |
| `cannot import name 'EvaluationService'` | Wrong content in file | Replace with correct EvaluationService class |
| `cannot import name 'F1Evaluator'` | Wrong content in f1.py | Replace with correct F1Evaluator class |
| Alembic `Server not found` | `\\` in .env | Use single `\` in .env |
| Alembic `table already exists` | Wrong database connected | Create new DB with sqlcmd, update .env |
| Ollama connection refused | Ollama not running | Run `ollama serve` in a separate terminal |
| Document stuck at Processing | Ollama not running at upload time | Ensure `ollama serve` is running before uploading |
| CORS error in browser | Frontend port wrong | Backend allows `http://localhost:5173` only |
| WebSocket not connecting | Backend not started | Start backend first |
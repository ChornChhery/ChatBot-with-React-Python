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
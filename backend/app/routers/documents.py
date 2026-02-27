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

    # FIX: No longer passing `db` — document_service.process_document now
    # creates its own fresh session internally to avoid session expiry issues
    background_tasks.add_task(document_service.process_document, doc.id, text, chunk_strategy)

    return {"id": doc.id, "file_name": doc.file_name, "message": "Upload started"}


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document_service.delete_document(db, document_id)
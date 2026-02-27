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
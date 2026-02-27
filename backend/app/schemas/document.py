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
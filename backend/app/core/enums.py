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
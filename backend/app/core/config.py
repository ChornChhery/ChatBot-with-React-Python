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
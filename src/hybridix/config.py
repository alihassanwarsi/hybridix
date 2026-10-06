from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 2000
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRIDIX_",
    )

settings = Settings()
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    brain_db_path: str = "./brain.db"
    brain_embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()

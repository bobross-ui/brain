from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    brain_db_path: str = "./brain.db"
    brain_embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "qwen2.5:3b-instruct"
    llm_provider: str = "ollama"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"


settings = Settings()

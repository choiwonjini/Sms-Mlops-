from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Google Cloud & Vertex AI
    GCP_PROJECT_ID: str
    GCP_LOCATION: str
    GOOGLE_APPLICATION_CREDENTIALS: str
    
    # Model Configuration
    MODEL_NAME: str
    LLM_PROVIDER: str
    
    # Execution Mode
    EXECUTION_MODE: str  # local, server
    DEBUG: bool
    MAX_TOOL_TURNS: int
    OCR_MODEL_NAME: str
    OCR_ENDPOINT_ID: str

    # API Server Configuration
    API_HOST: str
    API_PORT: int
    
    # Application Paths
    GUIDES_DIR: str

    # Message Language
    LANGUAGE: str

    # Store Info
    BANK_ACCOUNT_INFO: str
    PRICE_MODEL_NAME: str



    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

import os
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS


from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentType(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LLMProviderType(StrEnum):
    GROQ = "groq"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


    neo4j_uri: str = Field(alias="NEO4J_URI")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_username: str = Field(alias="NEO4J_USERNAME")
    neo4j_password: SecretStr = Field(alias="NEO4J_PASSWORD")

    floranexus_env: EnvironmentType = Field(default=EnvironmentType.DEV, alias="FLORANEXUS_ENV")

    floranexus_llm_provider: LLMProviderType = Field(default=LLMProviderType.GROQ, alias="FLORANEXUS_LLM_PROVIDER")
    floranexus_llm_base_url: str | None = Field(default=None, alias="FLORANEXUS_LLM_BASE_URL")
    floranexus_llm_api_key: SecretStr | None = Field(default=None, alias="FLORANEXUS_LLM_API_KEY")
    floranexus_llm_model: str = Field(default="llama-3.1-8b-instant", alias="FLORANEXUS_LLM_MODEL")
    floranexus_llm_temperature: float = Field(default=0.2, alias="FLORANEXUS_LLM_TEMPERATURE")
    floranexus_llm_timeout_s: int = Field(default=60, alias="FLORANEXUS_LLM_TIMEOUT_S")
    floranexus_llm_max_tool_rounds: int = Field(default=3, alias="FLORANEXUS_LLM_MAX_TOOL_ROUNDS")

    floranexus_langfuse_enabled: bool = Field(default=True, alias="FLORANEXUS_LANGFUSE_ENABLED")
    floranexus_prompt_behavior: str = Field(default="flora_behavior", alias="FLORANEXUS_PROMPT_BEHAVIOR")
    floranexus_prompt_tools: str = Field(default="flora_tools", alias="FLORANEXUS_PROMPT_TOOLS")
    floranexus_prompt_style: str = Field(default="flora_style_ro", alias="FLORANEXUS_PROMPT_STYLE")

    floranexus_confidence_high: float = Field(default=0.90, alias="FLORANEXUS_CONFIDENCE_HIGH")
    floranexus_confidence_medium: float = Field(default=0.75, alias="FLORANEXUS_CONFIDENCE_MEDIUM")

    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")

    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: SecretStr | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")

    @property
    def has_langfuse_credentials(self) -> bool:
        return bool(
            self.langfuse_secret_key
            and self.langfuse_public_key
            and self.langfuse_base_url
        )

    @property
    def effective_llm_api_key(self) -> str | None:
        if self.floranexus_llm_api_key:
            return self.floranexus_llm_api_key.get_secret_value()
        if self.floranexus_llm_provider == LLMProviderType.GROQ and self.groq_api_key:
            return self.groq_api_key.get_secret_value()
        return None

    @property
    def effective_llm_base_url(self) -> str | None:
        if self.floranexus_llm_base_url:
            return self.floranexus_llm_base_url.rstrip("/")
        if self.floranexus_llm_provider == LLMProviderType.OLLAMA:
            return "http://localhost:11434/api"
        return None


settings = Settings()  # type: ignore[call-arg]
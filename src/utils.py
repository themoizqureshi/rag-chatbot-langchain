"""Shared utilities: logging setup, environment validation, LLM factory."""

import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_api_key() -> None:
    """Raise EnvironmentError if neither GOOGLE_API_KEY nor OPENROUTER_API_KEY is set."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        raise EnvironmentError(
            "No LLM API key found. Set GOOGLE_API_KEY (https://aistudio.google.com/apikey) "
            "or OPENROUTER_API_KEY (https://openrouter.ai) in your .env file."
        )


def get_llm(temperature: float = 0):
    """
    Return a LangChain chat model.

    Prefers GOOGLE_API_KEY (Gemini). Falls back to OPENROUTER_API_KEY when
    the Google free-tier quota is exhausted — OpenRouter proxies the same
    Gemini 2.0 Flash model through an OpenAI-compatible endpoint.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openrouter_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

    raise EnvironmentError(
        "Set GOOGLE_API_KEY or OPENROUTER_API_KEY in your .env file."
    )

"""Token usage and cost tracking via LangChain callback."""

import logging
from dataclasses import dataclass, field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# USD per token (as of 2025-05). Gemini 2.0 Flash free tier has no cost,
# but we track as if paid so the numbers are meaningful for comparison.
_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.0-flash-001": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "google/gemini-2.0-flash-001": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    # Fallback for any unrecognised model — Gemini Flash pricing as a safe default
    "default": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}


@dataclass
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "unknown"

    def display(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out"
            f" | Est. cost: ${self.cost_usd:.6f}"
            f" | Model: {self.model}"
        )


class CostTracker(BaseCallbackHandler):
    """
    LangChain callback that captures token usage and estimated USD cost.

    Works with both Gemini (usage_metadata on AIMessage) and
    OpenRouter/OpenAI (token_usage in llm_output). Wire in at invoke/stream
    time via config={"callbacks": [tracker]} — no changes to chain needed.
    """

    def __init__(self) -> None:
        self._summary = UsageSummary()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:  # noqa: ANN003
        model = "default"

        # ── Gemini path: usage_metadata lives on the AIMessage ──────────────
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None) or {}
                self._summary.input_tokens += usage.get("input_tokens", 0)
                self._summary.output_tokens += usage.get("output_tokens", 0)
                self._summary.total_tokens += usage.get("total_tokens", 0)
                meta = getattr(msg, "response_metadata", {}) or {}
                model = meta.get("model_name", model)

        # ── OpenRouter / OpenAI path: token_usage in llm_output ─────────────
        llm_out = response.llm_output or {}
        oai_usage = llm_out.get("token_usage", {})
        if oai_usage:
            self._summary.input_tokens += oai_usage.get("prompt_tokens", 0)
            self._summary.output_tokens += oai_usage.get("completion_tokens", 0)
            self._summary.total_tokens += oai_usage.get("total_tokens", 0)
            model = llm_out.get("model_name", model)

        self._summary.model = model
        pricing = _PRICING.get(model, _PRICING["default"])
        self._summary.cost_usd = (
            self._summary.input_tokens * pricing["input"]
            + self._summary.output_tokens * pricing["output"]
        )
        logger.info("Cost tracker: %s", self._summary.display())

    @property
    def summary(self) -> UsageSummary:
        return self._summary

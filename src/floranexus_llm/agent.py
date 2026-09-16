from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from html import escape
from typing import Any, Iterator, Protocol

from groq import Groq

from floranexus_llm.flora_store import Neo4jFloraStore
from floranexus_llm.flora_tools import FloraToolsService, SessionState, ToolEvent
from floranexus_llm.prediction_payload import PredictionPayload
from floranexus_llm.settings import settings

try:
    from langfuse import get_client, propagate_attributes
except ImportError:  # pragma: no cover
    get_client = None
    propagate_attributes = None


MAX_HISTORY_TURNS = 2
MAX_COMPLETION_TOKENS = 280
DEFAULT_MAX_TOOL_ROUNDS = 4
MAX_TOOL_ITEMS_PER_CATEGORY = 3
MAX_TOOL_TEXT_CHARS = 220

_LANGFUSE_CLIENT = None


# ============================================================
# Generic settings helpers
# ============================================================

def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def get_provider_name() -> str:
    return _safe_str(getattr(settings, "floranexus_llm_provider", "groq"), "groq").lower()


def get_model_name() -> str:
    return _safe_str(getattr(settings, "floranexus_llm_model", ""))


def get_temperature() -> float:
    return float(getattr(settings, "floranexus_llm_temperature", 0.2))


def get_timeout_s() -> int:
    return int(getattr(settings, "floranexus_llm_timeout_s", 60))


def get_max_tool_rounds() -> int:
    return int(getattr(settings, "floranexus_llm_max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS))


def get_effective_llm_api_key() -> str | None:
    if hasattr(settings, "effective_llm_api_key"):
        return settings.effective_llm_api_key

    value = getattr(settings, "floranexus_llm_api_key", None)
    if value:
        return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)

    value = getattr(settings, "groq_api_key", None)
    if value:
        return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)

    return None


def get_effective_llm_base_url() -> str | None:
    if hasattr(settings, "effective_llm_base_url"):
        return settings.effective_llm_base_url

    value = getattr(settings, "floranexus_llm_base_url", None)
    if not value:
        return None
    return str(value).rstrip("/")


def is_langfuse_dev() -> bool:
    env_value = _safe_str(getattr(settings, "floranexus_env", "dev"), "dev").lower()
    return env_value == "dev"


# ============================================================
# Langfuse
# ============================================================

def _prepare_langfuse_env() -> None:
    public_key = getattr(settings, "langfuse_public_key", None)
    secret_key = getattr(settings, "langfuse_secret_key", None)
    base_url = getattr(settings, "langfuse_base_url", None)
    env_name = getattr(settings, "floranexus_env", "dev")

    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key.get_secret_value()
    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key.get_secret_value()
    if base_url:
        os.environ["LANGFUSE_BASE_URL"] = str(base_url)
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = str(env_name)


def get_langfuse_client():
    global _LANGFUSE_CLIENT

    enabled = bool(getattr(settings, "floranexus_langfuse_enabled", True))
    has_credentials = bool(getattr(settings, "has_langfuse_credentials", False))

    if not enabled:
        return None
    if not has_credentials:
        raise RuntimeError("Langfuse is enabled but credentials are missing from .env.")
    if get_client is None:
        raise RuntimeError("langfuse package is not installed.")

    _prepare_langfuse_env()

    if _LANGFUSE_CLIENT is None:
        _LANGFUSE_CLIENT = get_client()

    return _LANGFUSE_CLIENT


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()


@contextmanager
def agent_turn_trace(
    *,
    session_id: str,
    input_payload: dict[str, Any],
    user_message: str,
) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None or propagate_attributes is None:
        yield None
        return

    with client.start_as_current_observation(as_type="span", name="flora-agent-turn") as span:
        with propagate_attributes(session_id=session_id):
            span.update(
                input={"user_message": user_message, "payload": input_payload},
                metadata={
                    "session_id": session_id,
                    "request_id": input_payload.get("request_id"),
                    "status": input_payload.get("status"),
                },
            )
            yield span


@contextmanager
def tool_call_trace(*, tool_name: str, tool_arguments: dict[str, Any]) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(as_type="span", name=f"tool:{tool_name}") as span:
        span.update(input=tool_arguments, metadata={"tool_name": tool_name})
        yield span


@contextmanager
def llm_generation_trace(
    *,
    model_name: str,
    provider_name: str,
    messages: list[dict[str, Any]],
) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        as_type="generation",
        name=f"{provider_name}-chat-completion",
        model=model_name,
    ) as generation:
        generation.update(
            input=messages,
            metadata={"provider_name": provider_name},
        )
        yield generation


# ============================================================
# Prompt repository
# ============================================================

class PromptRepository:
    def _fetch_text_prompt(self, prompt_name: str) -> str:
        client = get_langfuse_client()

        try:
            if is_langfuse_dev():
                prompt = client.get_prompt(prompt_name, label="latest", cache_ttl_seconds=0)
            else:
                prompt = client.get_prompt(prompt_name)

            compiled = prompt.compile()
            if not isinstance(compiled, str) or not compiled.strip():
                raise RuntimeError(f"Prompt '{prompt_name}' is empty in Langfuse.")
            return compiled
        except Exception as exc:
            raise RuntimeError(f"Could not load prompt '{prompt_name}' from Langfuse: {exc}") from exc

    def get_system_prompt(self) -> str:
        behavior_name = _safe_str(getattr(settings, "floranexus_prompt_behavior", "flora_behavior"))
        tools_name = _safe_str(getattr(settings, "floranexus_prompt_tools", "flora_tools"))
        style_name = _safe_str(getattr(settings, "floranexus_prompt_style", "flora_style_ro"))

        behavior = self._fetch_text_prompt(behavior_name)
        tools = self._fetch_text_prompt(tools_name)
        style = self._fetch_text_prompt(style_name)
        return "\n\n".join([behavior, tools, style]).strip()


# ============================================================
# Helpers
# ============================================================

def xml_text(value: Any) -> str:
    return escape("" if value is None else str(value))


def trim_chat_history(history: list[dict[str, Any]], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, Any]]:
    if not history:
        return []

    max_messages = max_turns * 2
    trimmed = history[-max_messages:]

    cleaned: list[dict[str, Any]] = []
    for message in trimmed:
        role = message.get("role", "")
        content = _safe_str(message.get("content", ""))

        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue

        cleaned.append({"role": role, "content": content})

    return cleaned


def _short_text(value: Any, max_chars: int = MAX_TOOL_TEXT_CHARS) -> str:
    text = _safe_str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_prediction_context_xml(payload: PredictionPayload) -> str:
    if payload.status != "predicted":
        return """
<cnn_prediction_context>
  <status>waiting_for_prediction</status>
</cnn_prediction_context>
""".strip()

    top_k_rows = "\n".join(
        [
            f'    <candidate rank="{index}" plant_id="{xml_text(item.plant_id)}" score="{item.score:.4f}" />'
            for index, item in enumerate(payload.prediction.top_k, start=1)
        ]
    )

    return f"""
<cnn_prediction_context>
  <request_id>{xml_text(payload.request_id)}</request_id>
  <status>{xml_text(payload.status)}</status>
  <plant_id>{xml_text(payload.prediction.plant_id)}</plant_id>
  <plant_name_ro>{xml_text(payload.prediction.plant_name_ro)}</plant_name_ro>
  <plant_name_scientific>{xml_text(payload.prediction.plant_name_scientific)}</plant_name_scientific>
  <confidence>{payload.prediction.confidence:.4f}</confidence>
  <top_k>
{top_k_rows}
  </top_k>
</cnn_prediction_context>
""".strip()


def build_runtime_context_xml(session_state: SessionState) -> str:
    resolved = session_state.resolved_plant

    if resolved is None:
        return """
<runtime_context>
  <resolved_plant_status>missing</resolved_plant_status>
</runtime_context>
""".strip()

    aliases = ", ".join(resolved.aliases[:3]) if resolved.aliases else ""

    return f"""
<runtime_context>
  <resolved_plant_status>available</resolved_plant_status>
  <graph_plant_id>{xml_text(resolved.graph_plant_id)}</graph_plant_id>
  <common_name_ro>{xml_text(resolved.common_name_ro)}</common_name_ro>
  <scientific_name>{xml_text(resolved.scientific_name)}</scientific_name>
  <aliases>{xml_text(aliases)}</aliases>
</runtime_context>
""".strip()


def compact_tool_result(event: ToolEvent) -> dict[str, Any]:
    result = event.result

    if event.tool_name == "resolve_detected_plant":
        plant = result.get("plant", {})
        return {
            "resolved": result.get("resolved", False),
            "matched_by": result.get("matched_by", ""),
            "plant": {
                "graph_plant_id": plant.get("graph_plant_id", ""),
                "common_name_ro": plant.get("common_name_ro", ""),
                "scientific_name": plant.get("scientific_name", ""),
                "aliases": plant.get("aliases", [])[:3],
            },
        }

    if event.tool_name == "get_plant_identity":
        plant = result.get("plant", {})
        return {
            "found": result.get("found", False),
            "plant": {
                "graph_plant_id": plant.get("graph_plant_id", ""),
                "common_name_ro": plant.get("common_name_ro", ""),
                "scientific_name": plant.get("scientific_name", ""),
                "family": plant.get("family", ""),
                "aliases": plant.get("aliases", [])[:3],
            },
        }

    if event.tool_name == "get_plant_benefits":
        items = result.get("benefits", [])[:MAX_TOOL_ITEMS_PER_CATEGORY]
        return {
            "plant_name_ro": result.get("plant", {}).get("common_name_ro", ""),
            "benefits": [
                {
                    "text": _short_text(item.get("text", "")),
                    "benefit_type": item.get("benefit_type", ""),
                }
                for item in items
            ],
        }

    if event.tool_name == "get_plant_uses":
        items = result.get("uses", [])[:MAX_TOOL_ITEMS_PER_CATEGORY]
        return {
            "plant_name_ro": result.get("plant", {}).get("common_name_ro", ""),
            "uses": [
                {
                    "text": _short_text(item.get("text", "")),
                    "use_type": item.get("use_type", ""),
                }
                for item in items
            ],
        }

    if event.tool_name == "get_plant_contraindications":
        items = result.get("contraindications", [])[:MAX_TOOL_ITEMS_PER_CATEGORY]
        return {
            "plant_name_ro": result.get("plant", {}).get("common_name_ro", ""),
            "contraindications": [
                {
                    "category": item.get("category", ""),
                    "text": _short_text(item.get("text", "")),
                    "severity": item.get("severity", ""),
                }
                for item in items
            ],
        }

    if event.tool_name == "get_plant_interactions":
        items = result.get("interactions", [])[:MAX_TOOL_ITEMS_PER_CATEGORY]
        return {
            "plant_name_ro": result.get("plant", {}).get("common_name_ro", ""),
            "interactions": [
                {
                    "text": _short_text(item.get("text", "")),
                    "severity": item.get("severity", ""),
                }
                for item in items
            ],
        }

    if event.tool_name == "get_plant_usable_parts":
        items = result.get("usable_parts", [])[:MAX_TOOL_ITEMS_PER_CATEGORY]
        return {
            "plant_name_ro": result.get("plant", {}).get("common_name_ro", ""),
            "usable_parts": [
                {
                    "part_name_ro": item.get("part_name_ro", ""),
                    "notes_ro": _short_text(item.get("notes_ro", "")),
                }
                for item in items
            ],
        }

    if event.tool_name == "resolve_plant_candidates":
        items = result.get("candidates", [])[:3]
        return {
            "candidates": [
                {
                    "plant_name_ro": item.get("plant_name_ro", ""),
                    "plant_name_scientific": item.get("plant_name_scientific", ""),
                }
                for item in items
            ]
        }

    return {"result": _short_text(json.dumps(result, ensure_ascii=False))}


def build_assistant_tool_call_message(
    *,
    provider_name: str,
    content: str,
    tool_calls: list["NormalizedToolCall"],
) -> dict[str, Any]:
    if provider_name == "ollama":
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in tool_calls
        ],
    }


# ============================================================
# Tool calling DTOs
# ============================================================

@dataclass(slots=True)
class NormalizedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ChatCompletionResult:
    content: str
    tool_calls: list[NormalizedToolCall]


# ============================================================
# Provider clients
# ============================================================

class BaseChatClient(Protocol):
    provider_name: str
    model_name: str

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatCompletionResult:
        ...


class GroqChatClient:
    provider_name = "groq"

    def __init__(self) -> None:
        api_key = get_effective_llm_api_key()
        if not api_key:
            raise RuntimeError("Missing Groq API key.")

        self.client = Groq(
            api_key=api_key,
            timeout=get_timeout_s(),
        )
        self.model_name = get_model_name()

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatCompletionResult:
        with llm_generation_trace(
            model_name=self.model_name,
            provider_name=self.provider_name,
            messages=messages,
        ) as generation:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=get_temperature(),
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            message = response.choices[0].message
            content = _safe_str(message.content)

            normalized_calls: list[NormalizedToolCall] = []
            for raw_call in getattr(message, "tool_calls", None) or []:
                function = getattr(raw_call, "function", None)
                raw_arguments = getattr(function, "arguments", "{}") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}

                normalized_calls.append(
                    NormalizedToolCall(
                        id=_safe_str(getattr(raw_call, "id", "")),
                        name=_safe_str(getattr(function, "name", "")),
                        arguments=arguments,
                    )
                )

            if generation is not None:
                generation.update(
                    output={
                        "content": content,
                        "tool_calls": [asdict(call) for call in normalized_calls],
                    }
                )

            return ChatCompletionResult(content=content, tool_calls=normalized_calls)


class OllamaChatClient:
    provider_name = "ollama"

    def __init__(self) -> None:
        base_url = get_effective_llm_base_url()
        if not base_url:
            raise RuntimeError("Missing FLORANEXUS_LLM_BASE_URL for Ollama provider.")

        self.base_url = base_url
        self.api_key = get_effective_llm_api_key()
        self.model_name = get_model_name()
        self.timeout_s = get_timeout_s()

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatCompletionResult:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": get_temperature(),
                "num_predict": MAX_COMPLETION_TOKENS,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url=f"{self.base_url}/chat",
            data=body,
            headers=headers,
            method="POST",
        )

        with llm_generation_trace(
            model_name=self.model_name,
            provider_name=self.provider_name,
            messages=messages,
        ) as generation:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8")
                    parsed = json.loads(raw)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                debug_payload = json.dumps(payload, ensure_ascii=False)[:4000]
                raise RuntimeError(
                    f"Ollama HTTP error {exc.code}: {error_body}\n\nPayload preview:\n{debug_payload}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Could not reach Ollama provider: {exc}") from exc

            message = parsed.get("message", {}) or {}
            content = _safe_str(message.get("content", ""))

            normalized_calls: list[NormalizedToolCall] = []
            for index, raw_call in enumerate(message.get("tool_calls", []) or [], start=1):
                function = raw_call.get("function", {}) or {}
                arguments = function.get("arguments", {}) or {}
                if not isinstance(arguments, dict):
                    arguments = {}

                normalized_calls.append(
                    NormalizedToolCall(
                        id=_safe_str(raw_call.get("id", f"ollama-tool-{index}")),
                        name=_safe_str(function.get("name", "")),
                        arguments=arguments,
                    )
                )

            if generation is not None:
                generation.update(
                    output={
                        "content": content,
                        "tool_calls": [asdict(call) for call in normalized_calls],
                    }
                )

            return ChatCompletionResult(content=content, tool_calls=normalized_calls)


class ChatClientFactory:
    @staticmethod
    def create() -> BaseChatClient:
        provider = get_provider_name()

        if provider == "groq":
            return GroqChatClient()
        if provider == "ollama":
            return OllamaChatClient()

        raise RuntimeError(f"Unsupported provider: {provider}")


# ============================================================
# Agent
# ============================================================

class FloraNexusAgent:
    def __init__(self, store: Neo4jFloraStore) -> None:
        self.store = store
        self.tools = FloraToolsService(store)
        self.prompts = PromptRepository()
        self.llm = ChatClientFactory.create()

    def _trace_tool(self, event: ToolEvent) -> ToolEvent:
        with tool_call_trace(tool_name=event.tool_name, tool_arguments=event.arguments) as span:
            if span is not None:
                span.update(output=event.result)
        return event

    def _bootstrap_session(
        self,
        *,
        payload: PredictionPayload,
        session_state: SessionState,
    ) -> list[ToolEvent]:
        events: list[ToolEvent] = []

        if payload.status != "predicted":
            return events

        resolve_event = self.tools.ensure_resolved_plant(payload, session_state)
        if resolve_event is not None:
            events.append(self._trace_tool(resolve_event))

        return events

    def _build_first_turn_instruction(self, payload: PredictionPayload) -> str:
        if payload.status != "predicted":
            return (
                "Explică scurt și natural că încă nu există o predicție validă a plantei și că "
                "utilizatorul trebuie să trimită mai întâi rezultatul modelului de recunoaștere."
            )

        return (
            "Începe conversația natural pe baza predicției curente. "
            "Prezintă-te scurt, spune ce plantă a fost detectată, spune numele științific "
            "și alte denumiri dacă există. Nu intra încă în beneficii, utilizări sau contraindicații "
            "decât dacă utilizatorul le cere."
        )

    def _execute_model_tool_call(
        self,
        *,
        tool_call: NormalizedToolCall,
        payload: PredictionPayload,
        session_state: SessionState,
    ) -> ToolEvent:
        event = self.tools.execute_tool(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            payload=payload,
            session_state=session_state,
        )
        return self._trace_tool(event)

    def _direct_response_for_empty_tool(self, event: ToolEvent) -> str | None:
        if event.tool_name == "get_plant_benefits" and not event.result.get("benefits"):
            plant = event.result.get("plant", {}).get("common_name_ro", "această plantă")
            return f"În baza de date FloraNexus nu am găsit momentan beneficii pentru {plant}."

        if event.tool_name == "get_plant_uses" and not event.result.get("uses"):
            plant = event.result.get("plant", {}).get("common_name_ro", "această plantă")
            return f"În baza de date FloraNexus nu am găsit momentan informații despre utilizările pentru {plant}."

        if event.tool_name == "get_plant_contraindications" and not event.result.get("contraindications"):
            plant = event.result.get("plant", {}).get("common_name_ro", "această plantă")
            return f"În baza de date FloraNexus nu am găsit momentan contraindicații sau avertismente pentru {plant}."

        if event.tool_name == "get_plant_interactions" and not event.result.get("interactions"):
            plant = event.result.get("plant", {}).get("common_name_ro", "această plantă")
            return f"În baza de date FloraNexus nu am găsit momentan informații despre interacțiuni pentru {plant}."

        if event.tool_name == "get_plant_usable_parts" and not event.result.get("usable_parts"):
            plant = event.result.get("plant", {}).get("common_name_ro", "această plantă")
            return f"În baza de date FloraNexus nu am găsit momentan informații despre părțile utilizabile pentru {plant}."

        return None

    def run_turn(
        self,
        *,
        payload: PredictionPayload,
        user_message: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        session_state: SessionState | None = None,
    ) -> tuple[str, str, list[dict[str, Any]], SessionState, list[ToolEvent]]:
        current_session_id = session_id or f"flora-{os.urandom(8).hex()}"
        current_state = session_state or SessionState()
        stored_user_content = _safe_str(user_message)
        prior_history = trim_chat_history(chat_history or [], max_turns=MAX_HISTORY_TURNS)

        bootstrap_events = self._bootstrap_session(payload=payload, session_state=current_state)

        if payload.status != "predicted":
            direct_response = "Încă nu am primit o predicție validă a plantei. Trimite mai întâi rezultatul modelului de recunoaștere."
            new_chat_history = prior_history.copy()
            if stored_user_content:
                new_chat_history.append({"role": "user", "content": stored_user_content})
            new_chat_history.append({"role": "assistant", "content": direct_response})
            return current_session_id, direct_response, new_chat_history, current_state, bootstrap_events

        system_prompt = self.prompts.get_system_prompt()
        prediction_context_xml = build_prediction_context_xml(payload)
        runtime_context_xml = build_runtime_context_xml(current_state)
        tool_catalog_xml = self.tools.build_tool_catalog_xml()
        tool_schemas = self.tools.get_tool_schemas()

        effective_user_message = stored_user_content or self._build_first_turn_instruction(payload)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": prediction_context_xml},
            {"role": "system", "content": runtime_context_xml},
            {"role": "system", "content": tool_catalog_xml},
        ]
        messages.extend(prior_history)
        messages.append({"role": "user", "content": effective_user_message})

        all_tool_events = bootstrap_events.copy()

        with agent_turn_trace(
            session_id=current_session_id,
            input_payload=payload.model_dump(mode="python"),
            user_message=effective_user_message,
        ) as turn_span:
            for _ in range(get_max_tool_rounds()):
                completion = self.llm.complete(messages=messages, tools=tool_schemas)

                if completion.tool_calls:
                    messages.append(
                        build_assistant_tool_call_message(
                            provider_name=self.llm.provider_name,
                            content=completion.content or "",
                            tool_calls=completion.tool_calls,
                        )
                    )

                    current_round_events: list[ToolEvent] = []
                    for call in completion.tool_calls:
                        event = self._execute_model_tool_call(
                            tool_call=call,
                            payload=payload,
                            session_state=current_state,
                        )
                        all_tool_events.append(event)
                        current_round_events.append(event)

                        compact = compact_tool_result(event)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": json.dumps(compact, ensure_ascii=False),
                            }
                        )

                    if len(current_round_events) == 1:
                        direct_response = self._direct_response_for_empty_tool(current_round_events[0])
                        if direct_response is not None:
                            new_chat_history = prior_history.copy()
                            if stored_user_content:
                                new_chat_history.append({"role": "user", "content": stored_user_content})
                            new_chat_history.append({"role": "assistant", "content": direct_response})

                            if turn_span is not None:
                                turn_span.update(
                                    output={"assistant_text": direct_response},
                                    metadata={"tool_event_count": len(all_tool_events), "direct_response": True},
                                )

                            return current_session_id, direct_response, new_chat_history, current_state, all_tool_events

                    continue

                assistant_text = _safe_str(completion.content)
                if not assistant_text:
                    assistant_text = "Nu am putut formula momentan un răspuns."

                if turn_span is not None:
                    turn_span.update(
                        output={"assistant_text": assistant_text},
                        metadata={"tool_event_count": len(all_tool_events), "direct_response": False},
                    )

                new_chat_history = prior_history.copy()
                if stored_user_content:
                    new_chat_history.append({"role": "user", "content": stored_user_content})
                new_chat_history.append({"role": "assistant", "content": assistant_text})

                return current_session_id, assistant_text, new_chat_history, current_state, all_tool_events

        fallback_text = "Nu am putut finaliza răspunsul în limita de pași a agentului."
        new_chat_history = prior_history.copy()
        if stored_user_content:
            new_chat_history.append({"role": "user", "content": stored_user_content})
        new_chat_history.append({"role": "assistant", "content": fallback_text})
        return current_session_id, fallback_text, new_chat_history, current_state, all_tool_events


# ============================================================
# Interactive local test
# ============================================================

def interactive_chat(response_data: any) -> None:
    """
    Lansează chat-ul interactiv folosind datele primite direct de la client.
    :param response_data: Poate fi un dict (recomandat) sau un string JSON.
    """
    debug_tools = _safe_str(os.getenv("FLORANEXUS_DEBUG_TOOL_EVENTS", "false"), "false").lower() == "true"

    # Verificăm dacă datele vin ca string și le convertim în dict
    if isinstance(response_data, str):
        data = json.loads(response_data)
    else:
        data = response_data

    # Validăm payload-ul direct din variabilă, fără să mai atingem discul
    payload = PredictionPayload.model_validate(data)

    print(f"--- TERMINAL: Sesiune chat inițiată pentru {payload.prediction.plant_name_ro} ---")

    store = Neo4jFloraStore()
    store.verify_connectivity()

    agent = FloraNexusAgent(store=store)

    session_id: str | None = None
    chat_history: list[dict[str, Any]] = []
    session_state = SessionState()

    try:
        print("\n=== FloraNexus agent ===")
        print(f"Provider LLM: {get_provider_name()}")
        print(f"Model LLM: {agent.llm.model_name}")

        if payload.status == "predicted":
            print(
                f"Predicție încărcată: {payload.prediction.plant_name_ro} "
                f"({payload.prediction.plant_name_scientific}) | confidence={payload.prediction.confidence:.4f}"
            )
        else:
            print("Nu există încă o predicție validă în latest_prediction.json.")

        session_id, assistant_text, chat_history, session_state, tool_events = agent.run_turn(
            payload=payload,
            user_message=None,
            chat_history=chat_history,
            session_id=session_id,
            session_state=session_state,
        )

        print("\nAssistant:")
        print(assistant_text)

        if debug_tools:
            print("\nTool events:")
            print(json.dumps([asdict(event) for event in tool_events], indent=2, ensure_ascii=False))

        while True:
            user_text = input("\nTu > ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit", "q"}:
                break

            session_id, assistant_text, chat_history, session_state, tool_events = agent.run_turn(
                payload=payload,
                user_message=user_text,
                chat_history=chat_history,
                session_id=session_id,
                session_state=session_state,
            )

            print("\nAssistant:")
            print(assistant_text)

            if debug_tools:
                print("\nTool events:")
                print(json.dumps([asdict(event) for event in tool_events], indent=2, ensure_ascii=False))

    finally:
        flush_langfuse()
        store.close()


if __name__ == "__main__":
    interactive_chat()
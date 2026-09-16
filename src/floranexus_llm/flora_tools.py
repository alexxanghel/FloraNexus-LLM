from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from floranexus_llm.prediction_payload import PredictionPayload
from floranexus_llm.flora_store import Neo4jFloraStore


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_provider_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolEvent:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(slots=True)
class ResolvedPlant:
    graph_plant_id: str
    common_name_ro: str
    scientific_name: str
    family: str
    notes_ro: str
    aliases: list[str]
    matched_by: str


@dataclass(slots=True)
class SessionState:
    resolved_plant: ResolvedPlant | None = None


def _optional_graph_plant_id_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "graph_plant_id": {
                "type": "string",
                "description": (
                    "Optional internal FloraNexus Plant identifier. "
                    "Use it only if you intentionally want a different plant than the current resolved plant."
                ),
            }
        },
        "additionalProperties": False,
    }


class FloraToolsService:
    def __init__(self, store: Neo4jFloraStore) -> None:
        self.store = store
        self.catalog: list[ToolDefinition] = [
            ToolDefinition(
                name="get_plant_identity",
                description=(
                    "Use this when the user asks what plant it is, asks for the scientific name, aliases, family, "
                    "or other identity details. Returns grounded identity information for the current resolved plant."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="get_plant_benefits",
                description=(
                    "Use this when the user asks about benefits, what the plant helps with, positive effects, "
                    "or health-related benefits. Returns grounded Benefit facts from FloraNexus."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="get_plant_uses",
                description=(
                    "Use this when the user asks how the plant is used, what it is used for, "
                    "or asks about practical/traditional use. Returns grounded Use facts from FloraNexus."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="get_plant_contraindications",
                description=(
                    "Use this for contraindications, warnings, safety, adverse effects, risks, "
                    "or who should avoid the plant. Returns grounded Contraindication, Warning, and AdverseEffect data."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="get_plant_interactions",
                description=(
                    "Use this when the user asks about interactions with treatments, substances, or other plants. "
                    "Returns grounded Interaction data."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="get_plant_usable_parts",
                description=(
                    "Use this when the user asks which parts of the plant are used or usable. "
                    "Returns grounded PartUsed data."
                ),
                parameters=_optional_graph_plant_id_schema(),
            ),
            ToolDefinition(
                name="resolve_plant_candidates",
                description=(
                    "Use this when the user doubts the current prediction and asks for alternatives or likely candidates. "
                    "Returns the candidate plants from the current top_k list in readable form."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "plant_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional explicit list of candidate plant ids. "
                                "If omitted, use the current prediction top_k list."
                            ),
                        }
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.as_provider_tool() for tool in self.catalog]

    def build_tool_catalog_xml(self) -> str:
        lines = ["<available_tools>"]
        for tool in self.catalog:
            lines.append(
                f'  <tool name="{tool.name}">{tool.description}</tool>'
            )
        lines.append("</available_tools>")
        return "\n".join(lines)

    def resolve_detected_plant_from_payload(self, payload: PredictionPayload, session_state: SessionState) -> ToolEvent:
        event = ToolEvent(
            tool_name="resolve_detected_plant",
            arguments={
                "predicted_plant_id": payload.prediction.plant_id,
                "predicted_name_ro": payload.prediction.plant_name_ro,
                "predicted_scientific_name": payload.prediction.plant_name_scientific,
                "top_k_ids": [item.plant_id for item in payload.prediction.top_k],
            },
            result=self.store.resolve_detected_plant(
                predicted_plant_id=payload.prediction.plant_id,
                predicted_name_ro=payload.prediction.plant_name_ro,
                predicted_scientific_name=payload.prediction.plant_name_scientific,
                top_k_ids=[item.plant_id for item in payload.prediction.top_k],
            ),
        )

        result = event.result
        plant = result.get("plant", {})
        if result.get("resolved"):
            session_state.resolved_plant = ResolvedPlant(
                graph_plant_id=str(plant.get("graph_plant_id", "")),
                common_name_ro=str(plant.get("common_name_ro", "")),
                scientific_name=str(plant.get("scientific_name", "")),
                family=str(plant.get("family", "")),
                notes_ro=str(plant.get("notes_ro", "")),
                aliases=list(plant.get("aliases", [])),
                matched_by=str(result.get("matched_by", "")),
            )

        return event

    def ensure_resolved_plant(self, payload: PredictionPayload, session_state: SessionState) -> ToolEvent | None:
        if session_state.resolved_plant is not None:
            return None
        return self.resolve_detected_plant_from_payload(payload, session_state)

    @staticmethod
    def _effective_graph_plant_id(arguments: dict[str, Any], session_state: SessionState) -> str:
        explicit = str(arguments.get("graph_plant_id", "") or "").strip()
        if explicit:
            return explicit
        if session_state.resolved_plant is None:
            raise RuntimeError("Plant is not resolved in session state.")
        return session_state.resolved_plant.graph_plant_id

    def execute_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None,
        payload: PredictionPayload,
        session_state: SessionState,
    ) -> ToolEvent:
        args = arguments or {}

        if tool_name == "resolve_plant_candidates":
            plant_ids = args.get("plant_ids") or [item.plant_id for item in payload.prediction.top_k]
            return ToolEvent(
                tool_name="resolve_plant_candidates",
                arguments={"plant_ids": plant_ids},
                result=self.store.resolve_plant_candidates(plant_ids=plant_ids),
            )

        graph_plant_id = self._effective_graph_plant_id(args, session_state)

        if tool_name == "get_plant_identity":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_identity(graph_plant_id=graph_plant_id),
            )

        if tool_name == "get_plant_benefits":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_benefits(graph_plant_id=graph_plant_id),
            )

        if tool_name == "get_plant_uses":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_uses(graph_plant_id=graph_plant_id),
            )

        if tool_name == "get_plant_contraindications":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_contraindications(graph_plant_id=graph_plant_id),
            )

        if tool_name == "get_plant_interactions":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_interactions(graph_plant_id=graph_plant_id),
            )

        if tool_name == "get_plant_usable_parts":
            return ToolEvent(
                tool_name=tool_name,
                arguments={"graph_plant_id": graph_plant_id},
                result=self.store.get_plant_usable_parts(graph_plant_id=graph_plant_id),
            )

        raise RuntimeError(f"Unknown tool: {tool_name}")
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


PredictionStatus = Literal["waiting_for_prediction", "predicted"]
_NON_PRED_VALUES = {"", "none", "null", "n/a", "na"}


class PredictionInput(BaseModel):
    image_path: str
    source: str


class TopKItem(BaseModel):
    plant_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class PredictionBlock(BaseModel):
    plant_id: str = ""
    plant_name_ro: str = ""
    plant_name_scientific: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: list[TopKItem] = Field(default_factory=list)


class InferenceBlock(BaseModel):
    model_name: str
    model_version: str
    inference_ms: int = Field(ge=0)


class PredictionPayload(BaseModel):
    request_id: str
    timestamp: str
    input: PredictionInput
    prediction: PredictionBlock
    inference: InferenceBlock
    status: PredictionStatus

    @model_validator(mode="before")
    @classmethod
    def normalize_non_prediction_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        prediction = dict(data.get("prediction") or {})
        for key in ("plant_id", "plant_name_ro", "plant_name_scientific"):
            value = prediction.get(key)
            if isinstance(value, str) and value.strip().casefold() in _NON_PRED_VALUES:
                prediction[key] = ""

        status = data.get("status")
        if isinstance(status, str) and status.strip().casefold() in _NON_PRED_VALUES:
            data["status"] = "waiting_for_prediction"

        data["prediction"] = prediction
        return data

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "PredictionPayload":
        if self.status == "waiting_for_prediction":
            return self

        if not self.prediction.plant_id:
            raise ValueError("prediction.plant_id is required when status=predicted")
        if not self.prediction.plant_name_ro:
            raise ValueError("prediction.plant_name_ro is required when status=predicted")
        if not self.prediction.plant_name_scientific:
            raise ValueError("prediction.plant_name_scientific is required when status=predicted")
        if not self.prediction.top_k:
            raise ValueError("prediction.top_k is required when status=predicted")

        return self
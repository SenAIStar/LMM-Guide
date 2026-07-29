from dataclasses import dataclass
from typing import Any

from .contracts import validate_prediction


@dataclass(frozen=True)
class RouteResult:
    destination: str
    reasons: tuple[str, ...]


def route_prediction(prediction: dict[str, Any], media_count: int) -> RouteResult:
    errors = validate_prediction(prediction, media_count)
    if errors:
        return RouteResult("reject", tuple(errors))
    if prediction.get("decision") != "accept":
        return RouteResult("manual_review", (f"model decision is {prediction.get('decision')}",))
    return RouteResult("accept", ())

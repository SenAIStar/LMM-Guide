from typing import Any


VALID_ROLES = {"system", "user", "assistant"}


def validate_conversation_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for a multimodal conversation record."""
    errors: list[str] = []
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        errors.append("id must be a non-empty string")
    media = record.get("media")
    if not isinstance(media, list) or not media:
        errors.append("media must be a non-empty list")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant turns")
        return errors
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        if message.get("role") not in VALID_ROLES:
            errors.append(f"messages[{index}].role is invalid")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"messages[{index}].content must be non-empty")
    if messages[-1].get("role") != "assistant":
        errors.append("last message must be an assistant target")
    return errors


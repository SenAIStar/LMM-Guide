from __future__ import annotations

from typing import Any


def build_qdrant_filter(query: dict[str, Any]) -> Any:
    """Build a pre-retrieval payload filter with the current qdrant-client API."""
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("install requirements-store.txt") from exc
    must: list[Any] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=query["tenant_id"])),
        models.FieldCondition(
            key="acl", match=models.MatchAny(any=list(set(query["principals"]) | {"public"}))
        ),
        models.FieldCondition(
            key="effective_at", range=models.DatetimeRange(lte=query["as_of"])
        ),
    ]
    if query.get("product_scope"):
        must.append(
            models.FieldCondition(
                key="product_id", match=models.MatchValue(value=query["product_scope"])
            )
        )
    for name, value in (query.get("filters") or {}).items():
        must.append(models.FieldCondition(key=f"facts.{name}", match=models.MatchValue(value=value)))
    must_not = [models.FieldCondition(key="deleted", match=models.MatchValue(value=True))]
    # expires_at is stored only when a chunk expires; a second active flag is updated at ingestion time.
    must.append(models.FieldCondition(key="active", match=models.MatchValue(value=True)))
    return models.Filter(must=must, must_not=must_not)


def query_dense(client: Any, collection: str, vector: list[float], query: dict[str, Any], limit: int) -> list[Any]:
    return client.query_points(
        collection_name=collection,
        query=vector,
        using="dense",
        query_filter=build_qdrant_filter(query),
        with_payload=True,
        limit=limit,
    ).points


def switch_alias(client: Any, alias_name: str, old_collection: str | None, new_collection: str) -> None:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("install requirements-store.txt") from exc
    operations: list[Any] = []
    if old_collection:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=new_collection, alias_name=alias_name)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)

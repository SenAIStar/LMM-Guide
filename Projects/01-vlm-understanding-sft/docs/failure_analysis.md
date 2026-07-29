# Failure analysis record

For each failed sample, store:

- sample ID, group ID, split, and media SHA-256;
- dataset snapshot and license identifier;
- base model, processor, adapter, code, prompt, and schema versions;
- raw model text, parsed JSON, validation errors, gold label, and route;
- one actionable error bucket and the proposed data, model, prompt, or system change.

Use stable buckets: product-type confusion, color/lighting, invisible material, OCR, multi-view conflict, language-prior completion, unknown class, corrupt media, request limit, JSON/schema, and evidence mismatch. Do not collapse all errors into “hallucination”; object existence, unsupported attribute, and formatting failures require different fixes.

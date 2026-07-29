# Evaluation protocol

Freeze the test manifest, model revision, processor revision, prompt, pixel limits, decoding parameters, schema, and evaluator before comparing Zero-Shot with LoRA. Save one raw output and one parsed output for every sample.

The local evaluator reports only project-specific structured metrics. It does not implement or rename official COCO Caption, TextVQA, A-OKVQA, or POPE scores. Run those datasets with their official evaluation code and report them as separate diagnostics.

Required report slices: head and tail product types, single and multiple images, OCR/no-OCR, low resolution, glare, occlusion, unknown attributes, conflicting views, and unseen categories. Report missing predictions and review/reject rates with accuracy so selective systems cannot hide failures by abstaining.

All checked-in sample results are pipeline fixtures. `result_status=not_measured` remains authoritative until GPU predictions and a frozen evaluation report are stored.

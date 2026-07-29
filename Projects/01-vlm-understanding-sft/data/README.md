# Data layout

`dataset_info.json` resolves paths relative to the LLaMA-Factory `dataset_dir`. In this repository the demonstration files live under `data/sample/`, and image paths in JSONL are relative to `data/`.

The checked-in PPM is a generated format fixture, not a model-quality sample. It exists only so media hashing and path validation can run without downloading copyrighted or licensed datasets.

Before using ABO or another external dataset, store the download URL, snapshot date, archive hash, bundled license file, license hash, attribution text, and the human visual-label review decision. Do not copy listing metadata directly into the visual target.

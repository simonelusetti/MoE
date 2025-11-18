# Dataset Utilities

This directory contains scripts devoted to downloading or preparing raw datasets for exploratory analysis. Each script writes data in the most convenient on-disk format for that workflow, separate from the model-ready caches under `data/`.

## Available scripts

### `download_dataset.py`
- **Purpose**: Downloads raw Hugging Face datasets (any identifier/config) and saves the requested splits as `datasets` directories for offline work.
- **Output format**: `dataset_info.json` under the output folder, plus one subdirectory per split (e.g., `data/raw/wikiann/train/`) created with `Dataset.save_to_disk`.
- **Usage**:
  ```
  python tools/datasets/download_dataset.py --dataset ontonotes5 --config english_v4 --splits train validation test --output data/raw/ontonotes5_english_v4
  ```

### `build_dataset.py`
- **Purpose**: Builds model-ready caches by running `src.data.build_dataset` with a specified tokenizer/max length and saving the resulting `datasets` object under `data/*.pt`. When `--raw-root` is provided, the builder reads the raw splits from disk instead of pulling them from the Hugging Face Hub. (WNUT now requires this path because it no longer auto-downloads inside `src.data`.)
- **Output format**: Each invocation writes a `Dataset` to `data/<name>_<config>_<split>.pt/` using `save_to_disk`, containing precomputed encoder embeddings ready for training.
- **Usage**:
  ```
  python tools/datasets/build_dataset.py --dataset ontonotes --config english_v4 --splits train validation test --raw-root data/raw/ontonotes5_english_v4 --rebuild
  ```

Add new dataset utilities here following the same pattern, documenting the intended format and invocation.

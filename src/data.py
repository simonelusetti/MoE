# data.py
import ast
import os
import logging
import torch
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from dora import to_absolute_path
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

def _sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def _dataset_cache_filename(name, split, subset, cnn_field=None, dataset_config=None):
    parts = [name]
    if dataset_config:
        parts.append(_sanitize_fragment(dataset_config))
    if cnn_field:
        parts.append(cnn_field)
    parts.append(split)
    if subset is not None and subset != 1.0:
        parts.append(str(subset))
    return "_".join(parts) + ".pt"


def _freeze_encoder(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels=None):
    keep_labels = keep_labels or []

    def _tokenize_and_encode(x):
        # If we have ner_tags and tokens, align ner_tags to subword tokens
        has_ner = "ner_tags" in x and "tokens" in x
        if has_ner:
            enc = tok(x["tokens"], truncation=True, max_length=max_length, is_split_into_words=True)
        else:
            enc = tok(text_fn(x), truncation=True, max_length=max_length)

        device = next(encoder.parameters()).device
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"], device=device).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"], device=device).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs, output_attentions=True, return_dict=True)
            attns = out.attentions[-1].mean(1)   # last layer, avg heads [B,L,L]

            out_dict = {
                "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
                "embeddings": out.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32).numpy(),
                "incoming": attns.sum(-2).squeeze(0).detach().cpu().to(torch.float32).numpy(),   # [L]
                "outgoing": attns.sum(-1).squeeze(0).detach().cpu().to(torch.float32).numpy(),   # [L]
            }
            # Align ner_tags to subword tokens if present
            if has_ner:
                word_ids = enc.word_ids()
                ner_tags = x["ner_tags"]
                aligned_ner_tags = []
                for word_id in word_ids:
                    if word_id is None:
                        aligned_ner_tags.append(0)  # or -100 for ignore, but 0 = O
                    else:
                        aligned_ner_tags.append(ner_tags[word_id])
                out_dict["ner_tags"] = np.asarray(aligned_ner_tags, dtype=np.int64)
                for k in keep_labels:
                    if k not in ["ner_tags", "tokens"]:
                        out_dict[k] = x[k]
                # Optionally keep tokens for debugging
                out_dict["tokens"] = x["tokens"]
            else:
                for k in keep_labels:
                    out_dict[k] = x[k]
            return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)


FRAMENET_REPO_ID = "liyucheng/FrameNet_v17"
FRAMENET_CONFIG_FILES = {
    "fulltext": {
        "train": "fn1.7/fn1.7.fulltext.train.syntaxnet.conll",
        "validation": "fn1.7/fn1.7.dev.syntaxnet.conll",
        "test": "fn1.7/fn1.7.test.syntaxnet.conll",
    }
}


def _decode_token(raw_token: str) -> str:
    if raw_token.startswith("b'") and raw_token.endswith("'"):
        try:
            value = ast.literal_eval(raw_token)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")
        except (SyntaxError, ValueError):
            pass
    return raw_token


def _normalize_split_name(split: str) -> str:
    mapping = {
        "train": "train",
        "validation": "validation",
        "dev": "validation",
        "val": "validation",
        "test": "test",
    }
    try:
        return mapping[split]
    except KeyError as err:
        raise ValueError(f"Unsupported FrameNet split: {split!r}") from err


def _parse_framenet_conll_file(path: str):
    examples = {
        "tokens": [],
        "lemmas": [],
        "pos_tags": [],
        "lexical_units": [],
        "frame_elements": [],
        "frame_name": [],
    }

    tokens = []
    lemmas = []
    pos_tags = []
    lexical_units = []
    frame_elements = []
    frame_names_per_token = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    examples["tokens"].append(tokens)
                    examples["lemmas"].append(lemmas)
                    examples["pos_tags"].append(pos_tags)
                    examples["lexical_units"].append(lexical_units)
                    examples["frame_elements"].append(frame_elements)
                    frame_name = next((name for name in frame_names_per_token if name and name != "_"), "")
                    examples["frame_name"].append(frame_name)
                    tokens = []
                    lemmas = []
                    pos_tags = []
                    lexical_units = []
                    frame_elements = []
                    frame_names_per_token = []
                continue

            parts = stripped.split("\t")
            if len(parts) < 15:
                # Malformed line or metadata; skip gracefully.
                continue

            token = _decode_token(parts[1])
            lemma = parts[3]
            pos_tag = parts[5]
            lexical_unit = parts[12]
            frame_name = parts[13]
            frame_element = parts[14]

            tokens.append(token)
            lemmas.append(lemma)
            pos_tags.append(pos_tag)
            lexical_units.append(lexical_unit if lexical_unit != "_" else "")
            if not frame_element or frame_element == "_":
                frame_elements.append("O")
            else:
                frame_elements.append(frame_element)
            frame_names_per_token.append(frame_name if frame_name != "_" else "")

    if tokens:
        examples["tokens"].append(tokens)
        examples["lemmas"].append(lemmas)
        examples["pos_tags"].append(pos_tags)
        examples["lexical_units"].append(lexical_units)
        examples["frame_elements"].append(frame_elements)
        frame_name = next((name for name in frame_names_per_token if name and name != "_"), "")
        examples["frame_name"].append(frame_name)

    return examples


def _load_framenet_dataset(split: str, config_name: str):
    local_only = (
        os.environ.get("HF_HUB_OFFLINE", "").strip() == "1"
        or os.environ.get("HF_DATASETS_OFFLINE", "").strip() == "1"
    )
    normalized_split = _normalize_split_name(split)
    config_files = FRAMENET_CONFIG_FILES.get(config_name)
    if config_files is None:
        available = ", ".join(sorted(FRAMENET_CONFIG_FILES))
        raise ValueError(f"Unknown FrameNet config '{config_name}'. Available configs: {available}")
    filename = config_files.get(normalized_split)
    if filename is None:
        available = ", ".join(sorted(config_files))
        raise ValueError(
            f"Split '{split}' (normalized to '{normalized_split}') unsupported for FrameNet config '{config_name}'. "
            f"Available splits: {available}"
        )

    dataset_cache_root = os.environ.get("HF_DATASETS_CACHE")
    if not dataset_cache_root:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            dataset_cache_root = os.path.join(hf_home, "datasets")
    local_candidate = None
    if dataset_cache_root:
        local_candidate = os.path.join(dataset_cache_root, "liyucheng__FrameNet_v17", filename)
        if not os.path.exists(local_candidate):
            local_candidate = None
    if local_candidate is not None:
        local_path = local_candidate
    else:
        local_path = hf_hub_download(
            repo_id=FRAMENET_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_files_only=local_only,
        )
    parsed = _parse_framenet_conll_file(local_path)
    return Dataset.from_dict(parsed)


def build_dataset(
    name,
    split,
    tokenizer_name,
    max_length,
    subset=None,
    shuffle=False,
    cnn_field=None,
    dataset_config=None,
):
    """
    Generic dataset builder for CNN, WikiANN, CoNLL, and FrameNet.
    """
    # pick dataset + text extraction strategy
    if name == "cnn":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        if cnn_field is None: cnn_field = "highlights"
        text_fn = lambda x: x[cnn_field]
        keep_labels = []
    elif name == "wikiann":
        ds = load_dataset("wikiann", "en", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "conll2003":
        ds = load_dataset("conll2003", revision="refs/convert/parquet", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "framenet":
        config_name = dataset_config or "fulltext"
        ds = _load_framenet_dataset(split, config_name)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["tokens", "frame_elements", "frame_name", "lexical_units", "lemmas", "pos_tags"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    if shuffle:
        ds = ds.shuffle(seed=42)

    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = _freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels)
    return ds, tok

def initialize_dataloaders(cfg, logger):
    train_ds, _ = get_dataset(
        name=cfg.data.train.dataset,
        subset=cfg.data.train.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.train.shuffle,
        dataset_config=getattr(cfg.data.train, "config", None),
        cnn_field=getattr(cfg.data.train, "cnn_field", None),
    )
    eval_shuffle = bool(cfg.data.eval.shuffle)
    if eval_shuffle:
        logger.warning("Disabling shuffle for expert evaluation loader to preserve ordering.")
        eval_shuffle = False

    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=eval_shuffle,
        dataset_config=getattr(cfg.data.eval, "config", None),
        cnn_field=getattr(cfg.data.eval, "cnn_field", None),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.train.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.train.num_workers > 0),
        shuffle=cfg.data.train.shuffle,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.data.eval.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.eval.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.eval.num_workers > 0),
        shuffle=eval_shuffle,
    )
    return train_dl, eval_dl

# ---------- Collate ----------

from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # assume batch is a list of dicts
    def _as_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    input_ids = [_as_tensor(x["input_ids"], torch.long) for x in batch]
    attention_masks = [_as_tensor(x["attention_mask"], torch.long) for x in batch]
    incoming = [_as_tensor(x["incoming"], torch.float) for x in batch]
    outgoing = [_as_tensor(x["outgoing"], torch.float) for x in batch]

    has_ner = "ner_tags" in batch[0]
    if has_ner:
        ner_tags = [_as_tensor(x["ner_tags"], torch.long) for x in batch]

    # pad to longest sequence in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    incoming = pad_sequence(incoming, batch_first=True, padding_value=0.0)
    outgoing = pad_sequence(outgoing, batch_first=True, padding_value=0.0)
    if has_ner:
        ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-100)  # -100 is common ignore_index

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "incoming": incoming,
        "outgoing": outgoing
    }

    if has_ner:
        batch_out["ner_tags"] = ner_tags

    # add precomputed embeddings if your dataset already has them
    if "embeddings" in batch[0]:
        embeddings = [_as_tensor(x["embeddings"], torch.float) for x in batch]
        embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        batch_out["embeddings"] = embeddings

    return batch_out


# ---------- Loader ----------

def get_dataset(tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                name="cnn", split="train", max_length=256, dataset_config=None,
                cnn_field=None, subset=None, rebuild=False, shuffle=False):
    
    filename = _dataset_cache_filename(
        name,
        split,
        subset,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
    )
    path = to_absolute_path(f"./data/{filename}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if rebuild:
        raise RuntimeError("Dataset rebuilds are handled by tools/build_dataset.py. Run it before launching training.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset cache {path} not found. Run `tools/build_dataset.py --dataset {name} --splits {split}` to materialise it."
        )

    logger.info(f"Loading cached dataset from {path}")
    try:
        ds = load_from_disk(path)
    except (FileNotFoundError, ValueError) as err:
        raise RuntimeError("Dataset cache is unreadable. Rebuild it with tools/build_dataset.py.") from err

    if shuffle:
        ds = ds.shuffle(seed=42)

    return ds, tok

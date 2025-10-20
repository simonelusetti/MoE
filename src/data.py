# data.py
import os, logging, torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from dora import to_absolute_path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

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


def build_dataset(name, split, tokenizer_name, max_length, subset=None, shuffle=False, cnn_field=None):
    """
    Generic dataset builder for CNN, WikiANN, and CoNLL.
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
                name="cnn", split="train", max_length=256, cnn_field=None,
                subset=None, rebuild=False, shuffle=False):
    
    if cnn_field is not None:
        if subset is not None and subset != 1.0:
            path = f"./data/{name}_{cnn_field}_{split}_{subset}.pt"
        else:
            path = f"./data/{name}_{cnn_field}_{split}.pt"
    else:
        if subset is not None and subset != 1.0:
            path = f"./data/{name}_{split}_{subset}.pt"
        else:
            path = f"./data/{name}_{split}.pt"

    path = to_absolute_path(path)
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

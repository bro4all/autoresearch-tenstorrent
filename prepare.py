"""
One-time data preparation and runtime utilities for autoresearch-tenstorrent.

This file preserves the upstream tokenizer, dataloader packing strategy, and
bits-per-byte evaluation semantics while removing CUDA-only assumptions.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

from configs import load_config


cfg = load_config()

MAX_SEQ_LEN = cfg.max_seq_len
TIME_BUDGET = cfg.time_budget
EVAL_TOKENS = cfg.eval_tokens

CACHE_DIR = cfg.cache_dir
DATA_DIR = cfg.data_dir
TOKENIZER_DIR = cfg.tokenizer_dir
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| """
    r"""?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def _cfg():
    return load_config()


def ensure_cache_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def download_single_shard(index: int) -> bool:
    ensure_cache_dirs()
    filename = f"shard_{index:05d}.parquet"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return True
    url = f"{BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath.with_suffix(".parquet.tmp")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(temp_path, filepath)
            print(f"  downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  attempt {attempt}/5 failed for {filename}: {exc}")
            for path in (filepath, filepath.with_suffix(".parquet.tmp")):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
            time.sleep(2**attempt)
    return False


def download_data(num_shards: int, download_workers: int = 8) -> None:
    ensure_cache_dirs()
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if (DATA_DIR / f"shard_{i:05d}.parquet").exists())
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already present at {DATA_DIR}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")


def list_parquet_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    files = sorted(
        path for path in DATA_DIR.iterdir() if path.suffix == ".parquet" and not path.name.endswith(".tmp")
    )
    return files


def text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000) -> Iterator[str]:
    parquet_paths = [p for p in list_parquet_files() if p.name != VAL_FILENAME]
    nchars = 0
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_index)
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer() -> None:
    ensure_cache_dirs()
    tokenizer_pkl = TOKENIZER_DIR / "tokenizer.pkl"
    token_bytes_path = TOKENIZER_DIR / "token_bytes.pt"
    if tokenizer_pkl.exists() and token_bytes_path.exists():
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        raise RuntimeError("Need at least 2 data shards (1 train + 1 val) before tokenizer training")

    print("Tokenizer: training BPE tokenizer...")
    start = time.time()
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(
        text_iterator(),
        VOCAB_SIZE - len(SPECIAL_TOKENS),
        pattern=SPLIT_PATTERN,
    )
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    with tokenizer_pkl.open("wb") as handle:
        pickle.dump(encoding, handle)
    token_bytes = []
    special_set = set(SPECIAL_TOKENS)
    for token_id in range(encoding.n_vocab):
        token_str = encoding.decode([token_id])
        token_bytes.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), token_bytes_path)
    elapsed = time.time() - start
    print(f"Tokenizer: trained in {elapsed:.1f}s and saved to {tokenizer_pkl}")

    sample = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = encoding.encode_ordinary(sample)
    decoded = encoding.decode(encoded)
    assert decoded == sample, f"Tokenizer roundtrip failed: {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={encoding.n_vocab})")


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir: Path = TOKENIZER_DIR) -> "Tokenizer":
        with (tokenizer_dir / "tokenizer.pkl").open("rb") as handle:
            enc = pickle.load(handle)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads: int = 8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            rows = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in rows:
                    row.insert(0, prepend_id)
            return rows
        raise TypeError(f"Unsupported input type: {type(text)!r}")

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(list(ids))


def get_token_bytes(device=None) -> torch.Tensor:
    with (TOKENIZER_DIR / "token_bytes.pt").open("rb") as handle:
        token_bytes = torch.load(handle, map_location="cpu")
    if device is not None:
        return token_bytes.to(device)
    return token_bytes


def _document_batches(split: str, tokenizer_batch_size: int = 128) -> Iterator[Tuple[List[str], int]]:
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise RuntimeError("No parquet files found. Run prepare.py first.")
    val_path = DATA_DIR / VAL_FILENAME
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        if not parquet_paths:
            raise RuntimeError("No training shards found")
    elif split == "val":
        parquet_paths = [val_path]
    else:
        raise ValueError(f"Invalid split: {split}")
    epoch = 1
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for row_group_index in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_index)
                batch = row_group.column("text").to_pylist()
                for start in range(0, len(batch), tokenizer_batch_size):
                    yield batch[start : start + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device=None,
    buffer_size: int = 1000,
    tokenizer_batch_size: int | None = None,
):
    row_capacity = seq_len + 1
    batches = _document_batches(split, tokenizer_batch_size or _cfg().tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: List[List[int]] = []
    epoch = 1
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)

    def refill_buffer() -> None:
        nonlocal epoch
        docs, epoch = next(batches)
        doc_buffer.extend(tokenizer.encode(docs, prepend=bos_token))

    while True:
        for row_index in range(batch_size):
            position = 0
            while position < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - position
                best_index = -1
                best_length = 0
                for i, doc in enumerate(doc_buffer):
                    length = len(doc)
                    if length <= remaining and length > best_length:
                        best_index = i
                        best_length = length
                if best_index >= 0:
                    doc = doc_buffer.pop(best_index)
                    row_buffer[row_index, position : position + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    position += len(doc)
                else:
                    shortest_index = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_index)
                    row_buffer[row_index, position : position + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    position += remaining
        inputs = row_buffer[:, :-1].clone()
        targets = row_buffer[:, 1:].clone()
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(
    model,
    tokenizer: Tokenizer,
    batch_size: int,
    device=None,
    max_seq_len: int | None = None,
    eval_tokens: int | None = None,
) -> float:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    seq_len = max_seq_len or _cfg().max_seq_len
    num_eval_tokens = eval_tokens or _cfg().eval_tokens
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val", device=device)
    steps = max(1, num_eval_tokens // (batch_size * seq_len))
    total_nats = 0.0
    total_bytes = 0
    was_training = model.training
    model.eval()
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes.index_select(0, y_flat)
        mask = nbytes > 0
        total_nats += (loss * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if was_training:
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1))


def _write_parquet(path: Path, docs: Sequence[str]) -> None:
    table = pa.table({"text": list(docs)})
    pq.write_table(table, path)


def prepare_synthetic_cache(cache_dir: Path | None = None, seed: int = 123) -> None:
    target_root = cache_dir or _cfg().cache_dir
    data_dir = target_root / "data"
    tokenizer_dir = target_root / "tokenizer"
    if target_root.exists():
        shutil.rmtree(target_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    topics = ["tt", "autoresearch", "tokenizer", "causal lm", "rope"]
    verbs = ["packs", "scores", "trains", "predicts"]
    objects = ["tokens", "bytes", "sequences", "batches"]
    markers = ["alpha", "beta", "gamma", "delta"]

    def make_doc(i: int) -> str:
        pattern = i % 16
        topic = topics[pattern % len(topics)]
        pair = topics[(pattern + 1) % len(topics)]
        verb = verbs[pattern % len(verbs)]
        obj = objects[(pattern // 2) % len(objects)]
        marker = markers[(pattern // 3) % len(markers)]
        return (
            f"Document {pattern}. "
            f"{topic} model {verb} {obj}. "
            f"Pair {pair}. Marker {marker}. "
            f"Next token after topic {topic} is pair {pair}. "
            f"Repeat topic {topic}. Repeat verb {verb}. Repeat marker {marker}."
        )

    train_docs = [make_doc(i) for i in range(96)]
    val_docs = [make_doc(i + 8) for i in range(32)]
    _write_parquet(data_dir / "shard_00000.parquet", train_docs[:48])
    _write_parquet(data_dir / "shard_00001.parquet", train_docs[48:])
    _write_parquet(data_dir / VAL_FILENAME, val_docs)
    train_tokenizer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data/tokenizer for autoresearch-tenstorrent")
    parser.add_argument("--num-shards", type=int, default=_cfg().train_num_shards)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Use smoke-friendly data volume")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create a tiny synthetic cache instead of downloading upstream shards",
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    if args.synthetic or _cfg().synthetic_data:
        prepare_synthetic_cache(CACHE_DIR, seed=_cfg().seed)
        print("Prepared synthetic smoke cache")
        return

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    if args.smoke:
        num_shards = min(num_shards, max(2, _cfg().train_num_shards))
    download_data(num_shards, download_workers=args.download_workers)
    train_tokenizer()
    print("Done! Ready to train.")


if __name__ == "__main__":
    main()

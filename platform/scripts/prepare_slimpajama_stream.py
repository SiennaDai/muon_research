#!/usr/bin/env python3
"""Build a large SlimPajama token stream without holding tokens in RAM.

This is the low-memory counterpart of ``prepare_data.py --kind slimpajama``.
It produces the same on-disk contract: little-endian uint16 ``.bin`` splits and
a ``manifest.json`` with the identical schema, split algorithm, document list
and fingerprint derivation.  Tokens are appended to the split files as they are
encoded, so a 67M+ token training split fits in a few hundred MB of RAM.

Only the split capacities differ from the exploratory 4x dataset.  Because the
source revision, file order, document order, deduplication and split hashing are
unchanged, the validation and test streams are prefixes of the same documents
and should match the frozen 4x files byte for byte when the same caps are used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

SPLIT_ALGORITHM = "sha256(split-v1:content_hash) mod 10000; train<9800,val<9900,test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_files(api, dataset: str, revision: str, limit: int) -> list[dict]:
    entries = api.list_repo_tree(dataset, repo_type="dataset", revision=revision,
                                 recursive=True, expand=True)
    files = sorted(
        ({"path": x.path, "size": x.size} for x in entries
         if getattr(x, "size", None) and x.path.endswith(".parquet")),
        key=lambda item: item["path"])
    selected, total = [], 0
    for item in files:
        if total + item["size"] <= limit:
            selected.append(item)
            total += item["size"]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-max-bytes", type=int, default=6442450944)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--validation-tokens", type=int, default=65537)
    parser.add_argument("--test-tokens", type=int, default=65537)
    parser.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    import tiktoken

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    caps = {"train": args.train_tokens, "validation": args.validation_tokens,
            "test": args.test_tokens}

    api = HfApi()
    info = api.dataset_info(args.dataset, revision=args.source_revision)
    revision = info.sha
    selected = select_files(api, args.dataset, revision, args.source_max_bytes)
    plan = {"dataset": args.dataset, "revision": revision,
            "source_max_bytes": args.source_max_bytes,
            "selected_files": selected,
            "selected_bytes": sum(item["size"] for item in selected)}
    (output / "source_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not selected:
        raise SystemExit("no complete Parquet file fits source-max-bytes")

    enc = tiktoken.get_encoding("gpt2")
    rank_hash = hashlib.sha256()
    for token, rank in sorted(enc._mergeable_ranks.items()):
        rank_hash.update(token)
        rank_hash.update(rank.to_bytes(4, "little"))

    handles, digests, written = {}, {}, {}
    for split in caps:
        handles[split] = (output / f"{split}.bin.tmp").open("wb", buffering=1024 * 1024)
        digests[split] = hashlib.sha256()
        written[split] = 0
    docs, seen, pool = [], set(), []
    try:
        for source in selected:
            local = Path(hf_hub_download(args.dataset, source["path"],
                                         repo_type="dataset", revision=revision))
            pool.append({**source, "sha256": sha256_file(local)})
            parquet = pq.ParquetFile(local)
            for batch in parquet.iter_batches(columns=["text"], batch_size=64):
                for raw in batch.column(0).to_pylist():
                    text = (raw or "").replace("\r\n", "\n")
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    if content_hash in seen:
                        continue
                    seen.add(content_hash)
                    bucket = int(hashlib.sha256(("split-v1:" + content_hash).encode()).hexdigest(), 16) % 10000
                    split = "train" if bucket < 9800 else ("validation" if bucket < 9900 else "test")
                    if written[split] >= caps[split]:
                        continue
                    ids = enc.encode_ordinary(text) + [enc.eot_token]
                    payload = np.asarray(ids, dtype="<u2").tobytes()
                    handles[split].write(payload)
                    digests[split].update(payload)
                    written[split] += len(ids)
                    docs.append({"content_sha256": content_hash, "split": split,
                                 "tokens": len(ids)})
            print(f"[data] {source['path']} | written={written}", flush=True)
            if all(written[name] >= caps[name] for name in caps):
                break
    finally:
        for handle in handles.values():
            handle.flush()
            handle.close()
    if any(written[name] < caps[name] for name in caps):
        raise SystemExit(f"selected files did not supply target capacities: {written} vs {caps}")

    splits = {}
    for name in caps:
        tmp = output / f"{name}.bin.tmp"
        final = output / f"{name}.bin"
        tmp.replace(final)
        splits[name] = {"file": final.name, "tokens": int(written[name]),
                        "sha256": digests[name].hexdigest()}

    manifest = {
        "schema_version": 1,
        "protocol_id": "slimpajama-hash-split-v1",
        "source": {"dataset": args.dataset, "revision": revision},
        "file_pool": pool,
        "document_fingerprint": hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest(),
        "documents": docs,
        "split_algorithm": SPLIT_ALGORITHM,
        "tokenizer": {"name": "tiktoken:gpt2",
                      "version": importlib.metadata.version("tiktoken"),
                      "fingerprint": rank_hash.hexdigest()},
        "eot": enc.eot_token,
        "dtype": "uint16",
        "endianness": "little",
        "max_token_id": enc.eot_token,
        "splits": splits,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "splits": splits,
                      "fingerprint": manifest["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()

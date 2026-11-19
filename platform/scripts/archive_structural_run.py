#!/usr/bin/env python3
"""Sharded TAR archiving, SHA-256 manifests, GitHub release upload and pruning.

The flow used by the Kaggle runbook is:

1. ``archive``  -> write ``.tar`` shards and a manifest with per-file hashes.
2. ``upload``   -> publish the shards to a GitHub release and verify them.
3. ``prune``    -> delete archived local data once the release is verified.

Each shard is streamed to disk (no whole-run buffering) and kept below
``--shard-bytes``.  Pruning a structural chunk marks its ``index.json`` entry as
``pruned`` instead of removing the entry, so a later resume still knows the
collection extent and can continue appending chunks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_io import atomic_write_json  # noqa: E402

ARCHIVE_FORMAT = "structural_archive"
ARCHIVE_VERSION = 1
EXCLUDED_DIRS = {"checkpoints", "archive", "orphaned", "wandb"}
EXCLUDED_NAMES = {"structural_status.json", "snapshot_status.json", "snapshot_errors.jsonl"}


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def update_range(path: Path) -> tuple[int, int] | None:
    name = path.name
    if name.startswith("update_") and name.endswith(".pt"):
        try:
            value = int(name[len("update_"):-len(".pt")])
            return (value, value)
        except ValueError:
            return None
    if name.startswith("update_") and name.endswith(".manifest.json"):
        try:
            value = int(name[len("update_"):-len(".manifest.json")])
            return (value, value)
        except ValueError:
            return None
    if path.suffix == ".npz":
        try:
            with np.load(path) as archive:
                updates = archive["updates"]
                return (int(updates.min()), int(updates.max()))
        except Exception:
            return None
    return None


def iter_payload(run_dir: Path):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if set(relative.parts) & EXCLUDED_DIRS or path.name in EXCLUDED_NAMES:
            continue
        if path.name.endswith(".tmp") or path.name == "index.json":
            continue
        yield path, relative


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, limit: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.limit = limit
        self.shards: list[dict] = []
        self._handle = None
        self._tar = None
        self._members: list[dict] = []
        self._first = None
        self._last = None
        self._index = 0

    def _open(self) -> None:
        name = f"{self.prefix}-part{self._index:03d}.tar"
        self._path = self.out_dir / name
        self._handle = self._path.open("wb")
        self._tar = tarfile.open(fileobj=self._handle, mode="w", format=tarfile.PAX_FORMAT)
        self._members = []
        self._first = self._last = None

    def _close(self) -> None:
        if self._tar is None:
            return
        self._tar.close()
        self._handle.close()
        digest = sha256_file(self._path)
        self.shards.append({
            "file": self._path.name, "sha256": digest,
            "nbytes": int(self._path.stat().st_size),
            "members": list(self._members),
            "updates": (None if self._first is None else [self._first, self._last]),
        })
        self._index += 1
        self._tar = None

    def add(self, path: Path, relative: Path) -> dict:
        size = path.stat().st_size
        if self._tar is not None and self._handle.tell() + size + 4096 > self.limit:
            self._close()
        if self._tar is None:
            self._open()
        digest = sha256_file(path)
        ranges = update_range(path)
        self._tar.add(str(path), arcname=str(relative))
        member = {"path": str(relative), "sha256": digest, "nbytes": int(size),
                  "updates": None if ranges is None else list(ranges)}
        member["shard"] = f"{self.prefix}-part{self._index:03d}.tar"
        self._members.append(member)
        if ranges is not None:
            self._first = ranges[0] if self._first is None else min(self._first, ranges[0])
            self._last = ranges[1] if self._last is None else max(self._last, ranges[1])
        return member

    def close(self) -> None:
        self._close()


def build_archive(run_dir: Path, out_dir: Path, shard_bytes: int,
                  since: int | None, through: int | None) -> Path:
    run_id = run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir, run_id, shard_bytes)
    files = []
    coverage: dict[str, list[list[int]]] = {}
    for path, relative in iter_payload(run_dir):
        ranges = update_range(path)
        if ranges is not None and (
                (through is not None and ranges[0] > through)
                or (since is not None and ranges[1] < since)):
            continue
        member = writer.add(path, relative)
        files.append(member)
        if ranges is not None:
            group = relative.parts[0] if len(relative.parts) > 1 else "files"
            coverage.setdefault(group, []).append(list(ranges))
    writer.close()
    manifest = {
        "format": ARCHIVE_FORMAT, "version": ARCHIVE_VERSION,
        "run_id": run_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir.resolve()), "shard_bytes": int(shard_bytes),
        "since_update": since, "through_update": through,
        "verified": False, "release": None,
        "shards": writer.shards, "files": files,
        "coverage": {key: [min(r[0] for r in value), max(r[1] for r in value)]
                     for key, value in coverage.items()},
    }
    manifest_path = out_dir / f"{run_id}-archive-manifest.json"
    atomic_write_json(manifest_path, manifest)
    checksums = "\n".join(f"{shard['sha256']}  {shard['file']}"
                          for shard in writer.shards) + "\n"
    (out_dir / "SHA256SUMS").write_text(checksums)
    return manifest_path


# ---------------------------------------------------------------------------
# GitHub release upload / verification
# ---------------------------------------------------------------------------


def _requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise SystemExit("GitHub upload requires the 'requests' package") from exc


def _release(repo: str, tag: str, token: str) -> dict:
    requests = _requests()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{repo}/releases"
    response = requests.get(f"{base}/tags/{tag}", headers=headers, timeout=60)
    if response.status_code == 200:
        return response.json()
    created = requests.post(base, headers=headers, timeout=60, json={
        "tag_name": tag, "name": tag,
        "body": "Structural Emergence Study data archive"})
    if created.status_code >= 300:
        raise SystemExit(f"cannot create release {tag}: {created.status_code} {created.text}")
    return created.json()


def upload_manifest(manifest_path: Path, repo: str, tag: str, token: str,
                    verify_download: bool) -> Path:
    requests = _requests()
    manifest = json.loads(manifest_path.read_text())
    out_dir = manifest_path.parent
    release = _release(repo, tag, token)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    for shard in manifest["shards"]:
        path = out_dir / shard["file"]
        with path.open("rb") as handle:
            response = requests.post(
                f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets",
                headers={**headers, "Content-Type": "application/octet-stream"},
                params={"name": shard["file"]}, data=handle, timeout=3600)
        if response.status_code >= 300 and response.status_code != 422:
            raise SystemExit(f"upload failed for {shard['file']}: {response.status_code} {response.text}")
    assets = requests.get(
        f"https://api.github.com/repos/{repo}/releases/{release['id']}",
        headers=headers, timeout=60).json()["assets"]
    by_name = {asset["name"]: asset for asset in assets}
    verified = True
    for shard in manifest["shards"]:
        asset = by_name.get(shard["file"])
        if asset is None or int(asset["size"]) != int(shard["nbytes"]):
            verified = False
            continue
        digest = asset.get("digest")
        if digest:
            verified = verified and digest == f"sha256:{shard['sha256']}"
        elif verify_download:
            local = Path(tempfile.mkdtemp()) / shard["file"]
            with requests.get(asset["url"], headers={**headers, "Accept": "application/octet-stream"},
                              stream=True, timeout=3600) as response:
                response.raise_for_status()
                with local.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            verified = verified and sha256_file(local) == shard["sha256"]
            shutil.rmtree(local.parent, ignore_errors=True)
        else:
            verified = False
    manifest["verified"] = bool(verified)
    manifest["release"] = {"repo": repo, "tag": tag, "id": release["id"],
                           "url": release.get("html_url")}
    atomic_write_json(manifest_path, manifest)
    return manifest_path


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def prune(run_dir: Path, manifest_path: Path, *, require_verified: bool = True) -> int:
    manifest = json.loads(manifest_path.read_text())
    if require_verified and not manifest.get("verified"):
        raise SystemExit("refusing to prune: archive manifest is not verified")
    removed = 0
    archived_snapshots: set[int] = set()
    group_indexes: dict[str, dict] = {}
    for member in manifest["files"]:
        if member.get("updates") is None:
            continue  # never prune metadata / logs
        relative = Path(member["path"])
        if relative.parts and relative.parts[0] in EXCLUDED_DIRS:
            continue
        path = run_dir / relative
        group = relative.parts[0]
        if group == "snapshots" and relative.name.endswith(".pt"):
            try:
                archived_snapshots.add(int(relative.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        if group in group_indexes:
            index = group_indexes[group]
        else:
            index_path = run_dir / group / "index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text())
                group_indexes[group] = index
            else:
                index = None
        if index is not None:
            for entry in index.get("chunks", []):
                if entry["file"] == path.name and not entry.get("pruned"):
                    entry["pruned"] = True
                    entry["pruned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    atomic_write_json(run_dir / group / "index.json", index)
                    break
        if path.exists():
            path.unlink()
            removed += 1
    if archived_snapshots:
        ledger = run_dir / "snapshots" / "archived.json"
        existing = set()
        if ledger.is_file():
            existing = set(json.loads(ledger.read_text()).get("updates", []))
        atomic_write_json(ledger, {"updates": sorted(existing | archived_snapshots)})
    atomic_write_json(run_dir / "archive" / f"prune-{int(time.time())}.json", {
        "format": "structural_prune", "version": 1,
        "manifest": str(manifest_path), "removed_files": removed,
        "archive_sha256": sha256_file(manifest_path)})
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    archive = sub.add_parser("archive")
    archive.add_argument("--run-dir", required=True)
    archive.add_argument("--out-dir", required=True)
    archive.add_argument("--shard-bytes", type=int, default=1536 * 1024**2)
    archive.add_argument("--since-update", type=int)
    archive.add_argument("--through-update", type=int)

    upload = sub.add_parser("upload")
    upload.add_argument("--manifest", required=True)
    upload.add_argument("--repo", required=True)
    upload.add_argument("--tag", required=True)
    upload.add_argument("--token-env", default="GITHUB_TOKEN")
    upload.add_argument("--verify-download", action="store_true")

    prune_parser = sub.add_parser("prune")
    prune_parser.add_argument("--run-dir", required=True)
    prune_parser.add_argument("--manifest", required=True)
    prune_parser.add_argument("--allow-unverified", action="store_true")

    args = parser.parse_args()
    if args.command == "archive":
        path = build_archive(Path(args.run_dir), Path(args.out_dir),
                             args.shard_bytes, args.since_update, args.through_update)
        print(path)
    elif args.command == "upload":
        token = os.environ.get(args.token_env)
        if not token:
            raise SystemExit(f"environment variable {args.token_env} is not set")
        path = upload_manifest(Path(args.manifest), args.repo, args.tag, token,
                               args.verify_download)
        print(path)
    else:
        print(prune(Path(args.run_dir), Path(args.manifest),
                    require_verified=not args.allow_unverified))


if __name__ == "__main__":
    main()

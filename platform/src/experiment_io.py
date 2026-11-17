"""Run artifacts, event logs, environment capture and atomic checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Artifacts are deliberately strict JSON.  Callers must turn exceptional
    # floating-point outcomes into explicit, finite metadata rather than
    # serializing NaN/Infinity tokens.
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def environment_snapshot() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": cuda, "torch_cuda": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
    }


def _git_command(root: Path, args: tuple[str, ...], *, safe_directory: str | None = None):
    command = ["git"]
    if safe_directory is not None:
        command += ["-c", f"safe.directory={safe_directory}"]
    command += list(args)
    return subprocess.run(command, cwd=root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _ownership_error(stderr: str) -> bool:
    text = (stderr or "").lower()
    return "dubious ownership" in text or "safe.directory" in text


def _git_with_ownership_retry(root: Path, args: tuple[str, ...]):
    """Run one read-only git command, retrying only ownership failures.

    The retry is scoped to the single command (``-c safe.directory=*``) and
    never changes global git configuration.  The returned mode is recorded in
    ``source.json`` so a retry is never silent.
    """
    try:
        result = _git_command(root, args)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        return None, "git_unavailable", f"{type(exc).__name__}: {exc}", False
    if result.returncode == 0:
        return result, "ok", None, False
    ownership = _ownership_error(result.stderr)
    # ``git diff`` reports a dubious-ownership repository as "Not a git
    # repository" instead of the usual ownership message.  Retry once; if the
    # retry also fails it is classified as a genuine git error below.
    ambiguous_repository = "not a git repository" in (result.stderr or "").lower()
    if ownership or ambiguous_repository:
        retry = _git_command(root, args, safe_directory="*")
        if retry.returncode == 0:
            return retry, "ok_safe_directory", result.stderr.strip(), True
        return retry, "ownership_error" if ownership else "error", retry.stderr.strip() or result.stderr.strip(), True
    return result, "error", result.stderr.strip(), False


def source_snapshot(root: Path) -> dict:
    """Capture git provenance without making missing remotes fatal.

    Fields are ``None`` when the corresponding command cannot produce a value.
    ``provenance_status`` distinguishes an absent origin, an unavailable git
    binary, a read-only ownership retry, and other real git errors.  Git
    provenance failure must not abort training.
    """
    root = Path(root)
    errors: list[str] = []
    safe_directory_used = False
    hard_error = False
    unavailable = False

    remote_result, remote_mode, remote_error, remote_retry = _git_with_ownership_retry(root, ("remote", "get-url", "origin"))
    safe_directory_used = safe_directory_used or remote_retry
    remote = None
    remote_missing = False
    if remote_result is not None and remote_result.returncode == 0:
        remote = remote_result.stdout.strip()
    else:
        text = (remote_error or "").lower()
        remote_missing = "no such remote" in text or "does not appear to be a git repository" in text
        if remote_mode == "git_unavailable":
            unavailable = True
        elif not remote_missing:
            hard_error = True
        if remote_error:
            errors.append(f"remote: {remote_error}")

    commit_result, commit_mode, commit_error, commit_retry = _git_with_ownership_retry(root, ("rev-parse", "HEAD"))
    safe_directory_used = safe_directory_used or commit_retry
    upstream_commit = None
    no_commit = False
    if commit_result is not None and commit_result.returncode == 0:
        upstream_commit = commit_result.stdout.strip()
    else:
        text = (commit_error or "").lower()
        no_commit = ("unknown revision" in text or "needed a single revision" in text
                     or "does not have any commits yet" in text or "ambiguous argument" in text)
        if commit_mode == "git_unavailable":
            unavailable = True
        elif not no_commit:
            hard_error = True
        if commit_error:
            errors.append(f"commit: {commit_error}")

    status_result, status_mode, status_error, status_retry = _git_with_ownership_retry(root, ("status", "--porcelain"))
    safe_directory_used = safe_directory_used or status_retry
    dirty = None
    if status_result is not None and status_result.returncode == 0:
        dirty = bool(status_result.stdout.strip())
    else:
        if status_mode == "git_unavailable":
            unavailable = True
        else:
            hard_error = True
        if status_error:
            errors.append(f"status: {status_error}")

    diff_result, diff_mode, diff_error, diff_retry = _git_with_ownership_retry(
        root, ("diff", "--", "src", "scripts", "recipes", "tests", "docs"))
    safe_directory_used = safe_directory_used or diff_retry
    diff = None
    if diff_result is not None and diff_result.returncode == 0:
        diff = diff_result.stdout
    else:
        if diff_mode == "git_unavailable":
            unavailable = True
        else:
            hard_error = True
        if diff_error:
            errors.append(f"diff: {diff_error}")

    if unavailable:
        provenance_status = "git_unavailable"
    elif hard_error:
        provenance_status = "ownership_error" if safe_directory_used else "git_error"
    elif remote_missing:
        provenance_status = "remote_missing"
    elif no_commit:
        provenance_status = "no_commit"
    elif safe_directory_used:
        provenance_status = "ok_safe_directory"
    else:
        provenance_status = "ok"

    return {
        "remote": remote,
        "upstream_commit": upstream_commit,
        "dirty": dirty,
        "diff": diff,
        "provenance_status": provenance_status,
        "provenance_error": "; ".join(errors) if errors else None,
        "safe_directory_used": safe_directory_used,
    }


class EventWriter:
    def __init__(self, path: Path, run_id: str, segment_id: int, start_index: int = 0):
        self.path, self.run_id, self.segment_id, self.index = path, run_id, segment_id, start_index

    def write(self, event_type: str, completed_updates: int, processed_target_tokens: int, **values):
        event = {"schema_version": 1, "event_type": event_type, "run_id": self.run_id, "event_index": self.index, "completed_updates": completed_updates, "processed_target_tokens": processed_target_tokens, "segment_id": self.segment_id, **values}
        self.index += 1
        with self.path.open("a") as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def rng_state(cuda_used: bool) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if cuda_used else None}


def _cpu_byte_rng_state(value, name: str) -> torch.Tensor:
    """Normalize a serialized RNG state for PyTorch's CPU generator APIs.

    Checkpoints are normally written with CPU RNG-state tensors.  Loading a
    checkpoint with ``map_location='cuda:0'`` also relocates those tensors,
    however.  Both ``torch.set_rng_state`` and the per-device calls made by
    ``torch.cuda.set_rng_state_all`` consume CPU ByteTensors, so normalize
    every RNG state independently of the checkpoint's load location.
    """
    try:
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    except AttributeError as exc:
        raise TypeError(f"checkpoint RNG state {name} must be a torch tensor") from exc


def restore_rng(state: dict) -> None:
    torch_cpu = _cpu_byte_rng_state(state["torch_cpu"], "torch_cpu")
    torch_cuda = state["torch_cuda"]
    if torch_cuda is not None:
        torch_cuda = [_cpu_byte_rng_state(value, f"torch_cuda[{index}]") for index, value in enumerate(torch_cuda)]
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(torch_cpu)
    if torch_cuda is not None:
        torch.cuda.set_rng_state_all(torch_cuda)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of an entire file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    """Return the SHA-256 of a tensor's little-endian FP32 CPU bytes."""
    if not torch.is_tensor(value):
        raise TypeError("sha256_tensor expects a torch.Tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = tensor.numpy().astype("<f4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def fsync_file(path: str | Path) -> None:
    """Flush a file's contents to stable storage."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: str | Path) -> None:
    """Best-effort directory fsync; unsupported platforms are tolerated."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(Path(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_torch_save_with_digest(value, path: str | Path, *, fsync: bool = True) -> dict:
    """Atomically write a ``torch.save`` payload and return its digest.

    The temporary file lives in the destination directory so ``os.replace`` is
    guaranteed to stay on one filesystem.  The returned SHA-256 covers the
    entire serialized file, not the logical tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        digest = sha256_file(tmp)
        size = os.path.getsize(tmp)
        os.replace(tmp, path)
        if fsync:
            fsync_directory(path.parent)
        return {"path": str(path), "sha256": digest, "nbytes": size}
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, value, *, fsync: bool = True) -> None:
    """Atomically write strict JSON (no NaN/Infinity, no silent coercion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, path)
        if fsync:
            fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

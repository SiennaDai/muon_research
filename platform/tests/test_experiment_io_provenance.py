import shutil
import subprocess
from pathlib import Path

import pytest

import experiment_io


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    (root / "src").mkdir()
    (root / "src" / "file.txt").write_text("initial\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


def test_missing_origin_is_null_not_fatal(git_repo):
    snapshot = experiment_io.source_snapshot(git_repo)
    assert snapshot["remote"] is None
    assert snapshot["upstream_commit"] is not None
    assert len(snapshot["upstream_commit"]) == 40
    assert snapshot["dirty"] is False
    assert snapshot["diff"] == ""
    assert snapshot["provenance_status"] == "remote_missing"
    assert "No such remote" in snapshot["provenance_error"]


def test_commit_dirty_and_diff_are_still_collected(git_repo):
    (git_repo / "src" / "file.txt").write_text("changed\n")
    snapshot = experiment_io.source_snapshot(git_repo)
    assert snapshot["remote"] is None
    assert snapshot["upstream_commit"] is not None
    assert snapshot["dirty"] is True
    assert "changed" in snapshot["diff"]
    assert snapshot["provenance_status"] == "remote_missing"


def test_origin_is_recorded_when_present(git_repo):
    _git(git_repo, "remote", "add", "origin", "https://example.com/repo.git")
    snapshot = experiment_io.source_snapshot(git_repo)
    assert snapshot["remote"] == "https://example.com/repo.git"
    assert snapshot["provenance_status"] == "ok"
    assert snapshot["safe_directory_used"] is False


def test_non_repository_is_reported_without_raising(tmp_path):
    directory = tmp_path / "not-a-repo"
    directory.mkdir()
    snapshot = experiment_io.source_snapshot(directory)
    assert snapshot["upstream_commit"] is None
    assert snapshot["provenance_status"] != "ok"
    assert snapshot["provenance_error"]


class _Result:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_dubious_ownership_retries_with_scoped_safe_directory(monkeypatch):
    calls = []

    def fake_git(root, args, *, safe_directory=None):
        calls.append(safe_directory)
        if safe_directory is None:
            return _Result(128, "", "fatal: detected dubious ownership in repository")
        return _Result(0, "ok\n", "")

    monkeypatch.setattr(experiment_io, "_git_command", fake_git)
    result, mode, error, retried = experiment_io._git_with_ownership_retry(
        Path("."), ("status", "--porcelain"))
    assert result.returncode == 0
    assert mode == "ok_safe_directory"
    assert retried is True
    assert calls == [None, "*"]
    assert "dubious ownership" in error


def test_diff_not_a_repository_also_retries(monkeypatch):
    calls = []

    def fake_git(root, args, *, safe_directory=None):
        calls.append(safe_directory)
        if safe_directory is None:
            return _Result(129, "", "warning: Not a git repository. Use --no-index")
        return _Result(0, "diff\n", "")

    monkeypatch.setattr(experiment_io, "_git_command", fake_git)
    result, mode, error, retried = experiment_io._git_with_ownership_retry(
        Path("."), ("diff", "--", "src"))
    assert result.returncode == 0
    assert mode == "ok_safe_directory"
    assert retried is True
    assert calls == [None, "*"]

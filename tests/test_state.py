"""Tests for state management and snapshots."""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from agent_workspace.models import Snapshot
from agent_workspace.state import (
    list_snapshots,
    load_snapshot,
    save_snapshot,
    take_snapshot,
)
from agent_workspace.workspace import create_workspace


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )
    return repo


def test_take_snapshot(git_repo: Path) -> None:
    ws = create_workspace(git_repo, task="test snapshot")
    snap = take_snapshot(git_repo, ws.id)
    assert snap.workspace_id == ws.id
    assert snap.git_status is not None
    assert isinstance(snap.git_log, list)


@pytest.mark.parametrize("failed_command", ["status", "log"])
def test_take_snapshot_warns_on_git_failure(
    git_repo: Path, caplog: pytest.LogCaptureFixture, failed_command: str,
) -> None:
    ws = create_workspace(git_repo, task="failed snapshot")
    successful = {
        "status": "?? notes.txt\n",
        "log": "abc123 initial commit\n",
    }
    results = [
        subprocess.CompletedProcess(
            args=["git", command],
            returncode=128 if command == failed_command else 0,
            stdout="" if command == failed_command else successful[command],
            stderr="fatal: not a git repository\n" if command == failed_command else "",
        )
        for command in ("status", "log")
    ]

    with patch("agent_workspace.state.subprocess.run", side_effect=results):
        with caplog.at_level(logging.WARNING, logger="agent_workspace.state"):
            snap = take_snapshot(git_repo, ws.id)

    assert len(caplog.records) == 1
    assert f"git {failed_command} failed (rc=128): fatal: not a git repository" in caplog.text
    assert snap.git_status == {"porcelain": [] if failed_command == "status" else ["?? notes.txt"]}
    assert snap.git_log == ([] if failed_command == "log" else ["abc123 initial commit"])
    assert snap.untracked_files == ([] if failed_command == "status" else ["notes.txt"])


def test_take_snapshot_does_not_warn_on_success(
    git_repo: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    ws = create_workspace(git_repo, task="successful snapshot")
    with caplog.at_level(logging.WARNING, logger="agent_workspace.state"):
        snap = take_snapshot(git_repo, ws.id)
    assert snap.git_log
    assert not caplog.records


def test_save_and_load_snapshot(git_repo: Path) -> None:
    ws = create_workspace(git_repo, task="test save")
    snap = take_snapshot(git_repo, ws.id)
    path = save_snapshot(git_repo, snap)
    assert path.exists()

    loaded = load_snapshot(path)
    assert loaded.workspace_id == ws.id
    assert loaded.id == snap.id


def test_list_snapshots(git_repo: Path) -> None:
    ws = create_workspace(git_repo, task="test list")
    snap = take_snapshot(git_repo, ws.id)
    save_snapshot(git_repo, snap)

    snaps = list_snapshots(git_repo, ws.id)
    assert len(snaps) == 1


def test_list_snapshots_newest_first_when_uuid_order_differs(git_repo: Path) -> None:
    older = Snapshot(
        id="ffffffff",
        workspace_id="workspace",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )
    newer = Snapshot(
        id="00000000",
        workspace_id="workspace",
        timestamp=datetime(2026, 1, 2, 12, 0, 0),
    )

    save_snapshot(git_repo, older)
    save_snapshot(git_repo, newer)

    snaps = list_snapshots(git_repo, "workspace")
    assert [path.name for path in snaps] == [
        "00000000-20260102-120000.json",
        "ffffffff-20260101-120000.json",
    ]


def test_list_snapshots_empty(git_repo: Path) -> None:
    snaps = list_snapshots(git_repo, "nonexistent")
    assert snaps == []

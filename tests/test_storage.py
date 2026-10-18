from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from research_training.storage import ArtifactStore, file_sha256


class MemoryRemote:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploaded: list[str] = []

    def upload(self, local_path: Path, run_name: str, remote_name: str | None = None) -> str:
        name = remote_name or local_path.name
        destination = self.root / run_name / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        self.uploaded.append(name)
        return name

    def download_named(self, run_name: str, remote_name: str, local_path: Path) -> bool:
        source = self.root / run_name / remote_name
        if not source.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)
        return True

    def download_file_id(self, file_id: str, local_path: Path) -> None:
        source = self.root / "inputs" / file_id
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)


def test_versioned_remote_checkpoint_can_restore_on_fresh_machine(tmp_path: Path) -> None:
    remote = MemoryRemote(tmp_path / "remote")
    payload: dict[str, Any] = {"completed": False, "epoch": 2, "next_batch": 3}
    first = ArtifactStore(tmp_path / "machine-a", remote)
    first.save_checkpoint("run-1", payload, step=17)
    assert remote.uploaded == [
        "checkpoint_step_000000017_epoch_0002_batch_00003_progress.pt",
        "latest.json",
    ]

    restored = ArtifactStore(tmp_path / "machine-b", remote).restore_checkpoint("run-1")
    assert restored is not None
    assert torch.load(restored, weights_only=False) == payload


def test_missing_input_is_downloaded_by_id_and_verified(tmp_path: Path) -> None:
    remote = MemoryRemote(tmp_path / "remote")
    source = remote.root / "inputs" / "file-id"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable research input")
    local = tmp_path / "cache" / "dataset.bin"
    resolved = ArtifactStore(tmp_path / "outputs", remote).ensure_local(
        local, "file-id", file_sha256(source)
    )
    assert resolved.read_bytes() == source.read_bytes()

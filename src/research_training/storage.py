"""Atomic local artifacts and optional Google Drive synchronization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol


def atomic_replace(temporary: Path, destination: Path) -> None:
    """Atomically replace a file, tolerating short-lived Windows locks."""
    for attempt in range(5):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RemoteStore(Protocol):
    """Small remote-store surface used by the artifact manager."""

    def upload(self, local_path: Path, run_name: str, remote_name: str | None = None) -> str:
        """Upload a run artifact."""

    def download_file_id(self, file_id: str, local_path: Path) -> None:
        """Download an input artifact by provider file ID."""

    def download_named(self, run_name: str, remote_name: str, local_path: Path) -> bool:
        """Download a named artifact if it exists."""


class RcloneStore:
    """Google Drive storage through an existing private rclone login."""

    def __init__(self, remote: str, parent_folder_id: str, folder_path: str) -> None:
        if shutil.which("rclone") is None:
            raise RuntimeError("rclone is unavailable; install it or choose the api backend")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", remote):
            raise ValueError("Invalid rclone remote name")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", parent_folder_id):
            raise ValueError("Invalid Google Drive parent folder ID")
        parts = folder_path.strip("/").split("/")
        if any(part in {"", ".", ".."} or ":" in part for part in parts):
            raise ValueError("Invalid Google Drive folder path")
        self.root = f"{remote},root_folder_id={parent_folder_id}:"
        self.folder_path = "/".join(parts)

    @staticmethod
    def _run(*arguments: str, missing_ok: bool = False) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["rclone", *arguments], capture_output=True, text=True, check=False)
        if result.returncode and not missing_ok:
            raise RuntimeError(f"rclone failed: {result.stderr.strip()}")
        return result

    def _path(self, run_name: str, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
            raise ValueError("Invalid run name")
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Invalid artifact name")
        return f"{self.root}{self.folder_path}/{run_name}/{name}"

    def upload(self, local_path: Path, run_name: str, remote_name: str | None = None) -> str:
        """Copy a file to Google Drive."""
        name = remote_name or local_path.name
        self._run("copyto", str(local_path), self._path(run_name, name))
        return name

    def download_file_id(self, file_id: str, local_path: Path) -> None:
        """Download a Drive object by ID into an atomic local cache."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Invalid Google Drive file ID")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(local_path.name + ".download")
        try:
            remote = self.root.split(",root_folder_id=", maxsplit=1)[0] + ":"
            self._run("backend", "copyid", remote, file_id, str(temporary))
            atomic_replace(temporary, local_path)
        finally:
            temporary.unlink(missing_ok=True)

    def download_named(self, run_name: str, remote_name: str, local_path: Path) -> bool:
        """Download a named file, atomically, if present."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(local_path.name + ".download")
        try:
            result = self._run(
                "copyto", self._path(run_name, remote_name), str(temporary), missing_ok=True
            )
            if result.returncode:
                return False
            atomic_replace(temporary, local_path)
            return True
        finally:
            temporary.unlink(missing_ok=True)


class DriveApiStore:
    """Google Drive storage via Application Default Credentials."""

    def __init__(self, parent_folder_id: str, folder_path: str) -> None:
        try:
            import google.auth
            from googleapiclient.discovery import build
        except ImportError as error:
            raise RuntimeError("Install the [drive] extra for the api backend") from error
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        self.service: Any = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.parent_folder_id = parent_folder_id
        self.folder_path = folder_path.strip("/")

    def _find(self, parent: str, name: str) -> dict[str, str] | None:
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"'{parent}' in parents and name = '{escaped}' and trashed = false"
        files = (
            self.service.files()
            .list(q=query, fields="files(id,name,mimeType)")
            .execute()["files"]
        )
        if len(files) > 1:
            raise RuntimeError(f"Ambiguous Drive name: {name}")
        return files[0] if files else None

    def _folder(self, relative: str) -> str:
        parent = self.parent_folder_id
        for part in (self.folder_path + "/" + relative).split("/"):
            if not part:
                continue
            if part in {".", ".."}:
                raise ValueError("Invalid Drive folder path")
            match = self._find(parent, part)
            if match is None:
                match = self.service.files().create(
                    body={
                        "name": part,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent],
                    },
                    fields="id,mimeType",
                ).execute()
            if match["mimeType"] != "application/vnd.google-apps.folder":
                raise RuntimeError(f"Drive path component is a file: {part}")
            parent = str(match["id"])
        return parent

    def upload(self, local_path: Path, run_name: str, remote_name: str | None = None) -> str:
        """Create or update a Drive artifact."""
        from googleapiclient.http import MediaFileUpload

        parent = self._folder(run_name)
        name = remote_name or local_path.name
        existing = self._find(parent, name)
        media = MediaFileUpload(str(local_path), resumable=True)
        if existing:
            response = self.service.files().update(
                fileId=existing["id"], media_body=media, fields="id"
            ).execute()
        else:
            response = self.service.files().create(
                body={"name": name, "parents": [parent]}, media_body=media, fields="id"
            ).execute()
        return str(response["id"])

    def download_named(self, run_name: str, remote_name: str, local_path: Path) -> bool:
        """Download a named Drive artifact if present."""
        from googleapiclient.http import MediaIoBaseDownload

        match = self._find(self._folder(run_name), remote_name)
        if match is None:
            return False
        temporary = local_path.with_name(local_path.name + ".download")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            request = self.service.files().get_media(fileId=match["id"])
            with temporary.open("wb") as stream:
                downloader = MediaIoBaseDownload(stream, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            atomic_replace(temporary, local_path)
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def download_file_id(self, file_id: str, local_path: Path) -> None:
        """Download a Drive object by ID into an atomic local cache."""
        from googleapiclient.http import MediaIoBaseDownload

        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Invalid Google Drive file ID")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(local_path.name + ".download")
        try:
            request = self.service.files().get_media(fileId=file_id)
            with temporary.open("wb") as stream:
                downloader = MediaIoBaseDownload(stream, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            atomic_replace(temporary, local_path)
        finally:
            temporary.unlink(missing_ok=True)


class ArtifactStore:
    """Own local run directories and versioned remote checkpoints."""

    def __init__(self, output_root: Path, remote: RemoteStore | None = None) -> None:
        self.output_root = output_root
        self.remote = remote

    def run_dir(self, run_name: str) -> Path:
        """Return a stable run directory."""
        path = self.output_root / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_local(
        self, local_path: Path, file_id: str | None = None, expected_sha256: str | None = None
    ) -> Path:
        """Reuse or remotely fetch an immutable input, then verify its digest."""
        if not local_path.exists():
            if self.remote is None or not file_id:
                raise FileNotFoundError(f"Missing input artifact: {local_path}")
            self.remote.download_file_id(file_id, local_path)
        if expected_sha256 and file_sha256(local_path).lower() != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {local_path}")
        return local_path

    def save_checkpoint(self, run_name: str, payload: dict[str, Any], step: int) -> Path:
        """Save atomically, then publish a versioned remote snapshot and pointer."""
        import torch

        directory = self.run_dir(run_name)
        checkpoint = directory / "latest.pt"
        temporary = directory / "latest.pt.tmp"
        torch.save(payload, temporary)
        atomic_replace(temporary, checkpoint)
        if self.remote is not None:
            phase = "complete" if payload["completed"] else "progress"
            remote_name = (
                f"checkpoint_step_{step:09d}_epoch_{payload['epoch']:04d}"
                f"_batch_{payload['next_batch']:05d}_{phase}.pt"
            )
            self.remote.upload(checkpoint, run_name, remote_name)
            pointer = {"checkpoint": remote_name, "sha256": file_sha256(checkpoint), "step": step}
            pointer_path = directory / "latest.json"
            pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
            self.remote.upload(pointer_path, run_name)
        return checkpoint

    def upload_artifact(self, run_name: str, path: Path) -> None:
        """Publish a final metadata or weight artifact when remote storage is enabled."""
        if self.remote is not None:
            self.remote.upload(path, run_name)

    def restore_checkpoint(self, run_name: str) -> Path | None:
        """Use local state or fetch and verify the latest remote checkpoint."""
        directory = self.run_dir(run_name)
        checkpoint = directory / "latest.pt"
        if checkpoint.exists():
            return checkpoint
        if self.remote is None:
            return None
        pointer_path = directory / "latest.json"
        if not self.remote.download_named(run_name, "latest.json", pointer_path):
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        name = pointer.get("checkpoint")
        if not isinstance(name, str) or not name.startswith("checkpoint_step_"):
            raise ValueError("Invalid remote checkpoint pointer")
        if not self.remote.download_named(run_name, name, checkpoint):
            raise FileNotFoundError(f"Remote checkpoint is missing: {name}")
        if file_sha256(checkpoint) != pointer["sha256"]:
            checkpoint.unlink(missing_ok=True)
            raise ValueError("Downloaded checkpoint failed SHA-256 verification")
        return checkpoint

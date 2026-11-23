"""Download the pinned official PV-HSD-2025 archive and preserve source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

REPOSITORY = "MisakaMikoto128/PV-HSD-2025"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}?blobs=true"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "solar-ir-dataset-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/external/PV-HSD-2025"))
    parser.add_argument(
        "--revision", help="Pinned Hugging Face commit; defaults to current revision"
    )
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    metadata_url = API_URL
    if args.revision:
        metadata_url = (
            f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{args.revision}?blobs=true"
        )
    info_bytes = fetch(metadata_url)
    info = json.loads(info_bytes)
    revision = info["sha"]
    file_info = next(f for f in info["siblings"] if f["rfilename"] == "PV-HSD-2025.zip")
    expected_hash = file_info["lfs"]["sha256"]
    expected_size = file_info["size"]
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/PV-HSD-2025.zip"
    sources = root / "source_metadata"
    sources.mkdir(exist_ok=True)
    (sources / "huggingface_api.json").write_bytes(info_bytes)
    readme = fetch(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/README.md")
    (sources / "README.upstream.md").write_bytes(readme)
    archive = root / "PV-HSD-2025.zip"
    partial = root / "PV-HSD-2025.zip.part"
    if not archive.exists():
        print(f"Downloading revision {revision}: {expected_size:,} bytes", flush=True)
        # Cache-busting obtains a fresh signed URL if an intermediate redirect was cached.
        subprocess.run(
            [
                "curl",
                "--location",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                f"{url}?download=true&request_time={int(time.time())}",
            ],
            check=True,
        )
        candidate = partial
    else:
        candidate = archive
    print("Checking archive size and official LFS SHA-256...", flush=True)
    actual_hash = sha256(candidate)
    if candidate.stat().st_size != expected_size or actual_hash != expected_hash:
        raise ValueError(
            "Archive does not match official LFS size/SHA-256; retained for inspection"
        )
    if candidate != archive:
        partial.replace(archive)
    extracted = root / "extracted"
    marker = root / "extraction.complete.json"
    completed = json.loads(marker.read_text()) if marker.exists() else {}
    if not extracted.exists() or completed.get("archive_sha256") != actual_hash:
        if extracted.exists():
            raise ValueError(
                "Existing extracted data has no matching provenance; refusing overwrite"
            )
        staging = root / "extracted.partial"
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Archive symlink rejected: {member.filename}")
            unpacked_bytes = sum(member.file_size for member in members)
            if shutil.disk_usage(root).free < unpacked_bytes:
                raise ValueError("Insufficient space for archive extraction")
            print(f"Extracting {len(members):,} entries with ZIP CRC validation...", flush=True)
            z.extractall(staging)
        staging.replace(extracted)
        marker.write_text(json.dumps({"archive_sha256": actual_hash}, indent=2) + "\n")
    receipt = {
        "dataset": "PV-HSD-2025",
        "repository": REPOSITORY,
        "revision": revision,
        "dataset_page": f"https://huggingface.co/datasets/{REPOSITORY}",
        "download_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive),
        "archive_bytes": expected_size,
        "archive_sha256": actual_hash,
        "official_lfs_sha256": expected_hash,
        "upstream_readme_sha256": hashlib.sha256(readme).hexdigest(),
        "extracted_path": str(extracted),
        "usage_status": "downloaded_for_pretraining_evaluation_not_used_in_training",
        "transformations": [],
    }
    (root / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Download the raw data snapshot and verify its SHA-256 checksum.

Usage (from the project root):
    python -m src.data.download                 # verify existing file or download snapshot
    python -m src.data.download --force         # download again even if the file exists
    python -m src.data.download --allow-mismatch  # keep a file whose checksum differs
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import requests

from src.config import load_config, project_path

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
TIMEOUT_SECONDS = 60


def sha256sum(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Stream a file from url to target via a temporary .part file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
    partial.replace(target)
    logger.info("Saved to %s", target)


def verify(path: Path, expected: str) -> bool:
    """Compare the file checksum with the expected value and log the result."""
    actual = sha256sum(path)
    if actual == expected:
        logger.info("Checksum OK: %s", path.name)
        return True
    logger.error("Checksum mismatch for %s: expected %s, got %s", path.name, expected, actual)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="keep the file although its checksum differs from the config")
    args = parser.parse_args(argv)

    cfg = load_config()["data"]
    target = project_path(cfg["raw_file"])

    if target.exists() and not args.force:
        logger.info("File exists, skipping download: %s", target)
    elif cfg["snapshot_url"]:
        download(cfg["snapshot_url"], target)
    else:
        logger.error(
            "No snapshot_url configured. Download the CSV manually from %s "
            "and save it as %s",
            cfg["source_url"], target,
        )
        return 1

    if verify(target, cfg["sha256"]):
        return 0
    if args.allow_mismatch:
        logger.warning("Continuing with a different data state than the analysed snapshot.")
        return 0
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())

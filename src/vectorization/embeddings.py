"""Sentence embeddings of the lightly cleaned texts (input for KMeans), with a cache.

Encoding about 80,000 texts takes several minutes on a CPU. The result is saved
as .npy file together with a small JSON file. The JSON file holds a fingerprint
of the model and the texts: if either changes, the cache is not used.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Separator between texts in the fingerprint. It does not occur in the cleaned texts.
_SEPARATOR = "\x1f"


def encode(texts: Sequence[str], model_name: str, revision: str | None, batch_size: int) -> np.ndarray:
    """Encode texts with a sentence-transformers model.

    The vectors are L2-normalized: their dot product is the cosine similarity,
    and KMeans (Euclidean distance) then groups texts by cosine similarity.
    """
    # Imported here because loading torch takes a few seconds.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def fingerprint(texts: Sequence[str], model_name: str, revision: str | None) -> str:
    """SHA-256 over model name, revision and all texts in order."""
    digest = hashlib.sha256()
    digest.update(f"{model_name}@{revision}".encode("utf-8"))
    for text in texts:
        digest.update(_SEPARATOR.encode("utf-8"))
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def load_or_encode(
    texts: Sequence[str],
    model_name: str,
    revision: str | None,
    batch_size: int,
    cache_file: Path,
    encoder: Callable[..., np.ndarray] = encode,
) -> np.ndarray:
    """Return the embeddings from the cache if it matches, otherwise encode and cache them.

    `encoder` can be replaced in tests, so that no model has to be downloaded.
    """
    texts = list(texts)
    key = fingerprint(texts, model_name, revision)
    meta_file = cache_file.with_suffix(".json")

    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == key:
            logger.info("Embeddings loaded from cache %s", cache_file)
            return np.load(cache_file)
        logger.info("Cache %s is outdated (texts or model changed)", cache_file)

    embeddings = encoder(texts, model_name, revision, batch_size)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, embeddings)
    meta = {
        "fingerprint": key,
        "model": model_name,
        "revision": revision,
        "n_texts": len(texts),
        "dimensions": int(embeddings.shape[1]),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Embeddings saved to cache %s", cache_file)
    return embeddings

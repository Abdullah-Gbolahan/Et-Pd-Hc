"""
model_downloader.py — Downloads model weights from Google Drive using gdown.

Supports two separate model keys (T1 / T2) via secrets:
  GDRIVE_FILE_ID_T1 = "..."   # T1-weighted MRI model
  GDRIVE_FILE_ID_T2 = "..."   # T2-weighted MRI model

Legacy single-key fallback: GDRIVE_FILE_ID (maps to T1 slot).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/neuroscan_models"))

MODEL_FILENAMES = {
    "T1": "swin_classifier_t1.pth",
    "T2": "swin_classifier_t2.pth",
}

MIN_VALID_BYTES = 1_000_000  # 1 MB — anything smaller is likely an HTML error page


def _is_valid_checkpoint(path: Path) -> bool:
    """Return True only if the file exists, is large enough, and starts with
    a PyTorch pickle magic byte (0x80) rather than HTML ('<')."""
    if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    # Reject obvious HTML error pages; accept pickle (0x80) and ZIP/PyTorch 2.x (PK = 0x50 0x4B)
    return header[:1] not in (b"<", b"{")


def _get_file_id(mri_type: str) -> str | None:
    secret_key = f"GDRIVE_FILE_ID_{mri_type}"
    try:
        return st.secrets[secret_key]
    except (KeyError, FileNotFoundError):
        pass
    if mri_type == "T1":
        try:
            return st.secrets["GDRIVE_FILE_ID"]
        except (KeyError, FileNotFoundError):
            pass
    return None


def _diagnose(save_path: Path) -> str:
    """Return a human-readable diagnosis of what was downloaded."""
    if not save_path.exists():
        return "File was not created at all."
    size = save_path.stat().st_size
    with open(save_path, "rb") as f:
        header = f.read(256)
    header_hex    = header[:8].hex()
    header_text   = header[:256].decode("utf-8", errors="replace").replace("\n", " ")[:120]
    return (
        f"Size: {size:,} bytes | "
        f"First 8 bytes (hex): `{header_hex}` | "
        f"First 120 chars: `{header_text}`"
    )


def download_from_gdrive(file_id: str, save_path: Path, progress_bar=None) -> Path:
    """Download via gdown, which correctly handles Drive's virus-scan confirmation."""
    import gdown

    save_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=str(save_path), quiet=False)

    if not _is_valid_checkpoint(save_path):
        diagnosis = _diagnose(save_path)
        save_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid PyTorch checkpoint.\n\n"
            f"**Diagnosis:** {diagnosis}\n\n"
            "This usually means:\n"
            "1. The file is not shared as **Anyone with the link** on Google Drive\n"
            "2. The FILE_ID in your secrets is wrong\n"
            "3. Google Drive download quota has been exceeded for this file"
        )

    if progress_bar:
        progress_bar.progress(1.0)

    return save_path


def ensure_model_downloaded(mri_type: str = "T1") -> Path:
    """
    Called once at app startup via @st.cache_resource.
    Downloads the checkpoint if not already cached.
    Returns the local path to the .pth file.
    """
    filename  = MODEL_FILENAMES.get(mri_type, MODEL_FILENAMES["T1"])
    save_path = MODEL_CACHE_DIR / filename

    if _is_valid_checkpoint(save_path):
        return save_path

    if save_path.exists():
        save_path.unlink()

    file_id = _get_file_id(mri_type)
    if not file_id:
        secret_key = f"GDRIVE_FILE_ID_{mri_type}"
        st.error(
            f"**Missing `{secret_key}` in secrets.**\n\n"
            f"Add it to `.streamlit/secrets.toml`:\n"
            f"```toml\n{secret_key} = \"your_file_id_here\"\n```"
        )
        st.stop()

    st.info(f"⬇️ Downloading **{mri_type}** model weights from Google Drive (one-time ~300 MB)…")
    bar = st.progress(0.0)

    try:
        path = download_from_gdrive(file_id, save_path, progress_bar=bar)
        return path
    except Exception as e:
        st.error(str(e))
        st.stop()

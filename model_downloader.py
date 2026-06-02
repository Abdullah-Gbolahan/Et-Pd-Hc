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
    # PyTorch checkpoints are pickles → first byte is 0x80
    # HTML error pages start with '<' (0x3C)
    return header[:1] == b"\x80"


def _get_file_id(mri_type: str) -> str | None:
    secret_key = f"GDRIVE_FILE_ID_{mri_type}"
    try:
        return st.secrets[secret_key]
    except (KeyError, FileNotFoundError):
        pass
    # Legacy fallback for T1
    if mri_type == "T1":
        try:
            return st.secrets["GDRIVE_FILE_ID"]
        except (KeyError, FileNotFoundError):
            pass
    return None


def download_from_gdrive(file_id: str, save_path: Path, progress_bar=None) -> Path:
    """Download via gdown, which correctly handles Drive's virus-scan confirmation."""
    import gdown

    save_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    # gdown writes directly to save_path
    gdown.download(url, str(save_path), quiet=False)

    # Validate what we got
    if not _is_valid_checkpoint(save_path):
        # Remove the bad file so next boot retries
        save_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file does not look like a valid PyTorch checkpoint. "
            "Make sure the Google Drive file is shared with 'Anyone with the link' "
            "and that the FILE_ID is correct."
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

    # Happy path — already have a valid file
    if _is_valid_checkpoint(save_path):
        return save_path

    # Remove stale / corrupt file if present
    if save_path.exists():
        save_path.unlink()

    file_id = _get_file_id(mri_type)
    if not file_id:
        secret_key = f"GDRIVE_FILE_ID_{mri_type}"
        st.error(
            f"**Missing `{secret_key}` in secrets.**\n\n"
            f"Add it to `.streamlit/secrets.toml`:\n"
            f"```toml\n{secret_key} = \"your_file_id_here\"\n```\n\n"
            "See README for how to get the file ID from your Drive share link."
        )
        st.stop()

    st.info(f"⬇️ Downloading **{mri_type}** model weights from Google Drive (one-time ~300 MB)…")
    bar = st.progress(0.0)

    try:
        path = download_from_gdrive(file_id, save_path, progress_bar=bar)
        return path
    except Exception as e:
        st.error(
            f"**Download failed:** `{e}`\n\n"
            "Check that your Drive file is shared with **Anyone with the link** "
            "and that the FILE_ID in your secrets is correct."
        )
        st.stop()

"""
model_downloader.py — Downloads model weights from Google Drive.

Supports two separate model keys (T1 / T2) via secrets:
  GDRIVE_FILE_ID_T1 = "..."   # T1-weighted MRI model
  GDRIVE_FILE_ID_T2 = "..."   # T2-weighted MRI model

Legacy single-key fallback: GDRIVE_FILE_ID (maps to T1 slot).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
import streamlit as st

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/neuroscan_models"))
CHUNK_SIZE_MB   = 8

MODEL_FILENAMES = {
    "T1": "swin_classifier_t1.pth",
    "T2": "swin_classifier_t2.pth",
}


def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _confirm_token(session: requests.Session, url: str) -> str:
    r = session.get(url, stream=True, timeout=30)
    for key, val in r.cookies.items():
        if key.startswith("download_warning"):
            return f"{url}&confirm={val}"
    content = b""
    for chunk in r.iter_content(chunk_size=32768):
        content += chunk
        if len(content) > 1_000_000:
            break
    match = re.search(rb'confirm=([0-9A-Za-z_\-]+)', content)
    if match:
        return f"{url}&confirm={match.group(1).decode()}"
    match = re.search(rb'"downloadUrl":"(https://[^"]+)"', content)
    if match:
        return match.group(1).decode().replace(r"\u003d", "=").replace(r"\u0026", "&")
    return url


def download_from_gdrive(file_id: str, save_path: Path, progress_bar=None) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and save_path.stat().st_size > 1_000_000:
        return save_path

    session  = requests.Session()
    dl_url   = _confirm_token(session, _gdrive_url(file_id))

    with session.get(dl_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total      = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE_MB * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_bar and total:
                        progress_bar.progress(min(downloaded / total, 1.0))
    return save_path


def ensure_model_downloaded(mri_type: str = "T1") -> Path:
    """
    Download model for the given MRI type ("T1" or "T2").
    Reads GDRIVE_FILE_ID_T1 / GDRIVE_FILE_ID_T2 from secrets.
    Falls back to legacy GDRIVE_FILE_ID for T1.
    """
    filename  = MODEL_FILENAMES.get(mri_type, MODEL_FILENAMES["T1"])
    save_path = MODEL_CACHE_DIR / filename

    if save_path.exists() and save_path.stat().st_size > 1_000_000:
        return save_path

    secret_key = f"GDRIVE_FILE_ID_{mri_type}"
    try:
        file_id = st.secrets[secret_key]
    except (KeyError, FileNotFoundError):
        # Legacy fallback for T1
        if mri_type == "T1":
            try:
                file_id = st.secrets["GDRIVE_FILE_ID"]
            except (KeyError, FileNotFoundError):
                file_id = None
        else:
            file_id = None

    if not file_id:
        st.error(
            f"**Missing `{secret_key}` in secrets.**\n\n"
            f"Add it to `.streamlit/secrets.toml`:\n"
            f"```toml\n{secret_key} = \"your_file_id_here\"\n```\n\n"
            "See README for how to get your Drive file ID."
        )
        st.stop()

    st.info(f"⬇️ Downloading **{mri_type}** model weights from Google Drive (one-time ~300 MB)…")
    bar  = st.progress(0.0)
    try:
        path = download_from_gdrive(file_id, save_path, progress_bar=bar)
        bar.progress(1.0)
        return path
    except Exception as e:
        st.error(
            f"Download failed: `{e}`\n\n"
            "Check that your Drive file is shared with **Anyone with the link**."
        )
        st.stop()

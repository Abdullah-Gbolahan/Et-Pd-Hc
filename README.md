# 🧠 NeuroScan AI — 3-Class Brain MRI Classifier

> **Research prototype · Not for clinical use**

Streamlit deployment of a Swin Transformer (Swin-B or Swin-S) fine-tuned to classify brain MRI slices into three classes: **ET** (Essential Tremor), **PD** (Parkinson's Disease), and **Healthy**. Supports both **T1-weighted** and **T2-weighted** MRI models, native **DICOM (.dcm)** input, GradCAM and Attention Rollout explainability maps, brain masking, and optional TTA.

---

## 📁 File Structure

```
├── app.py                        # Streamlit UI
├── inference.py                  # Model definition, prediction, GradCAM, Rollout
├── model_downloader.py           # Auto-downloads T1/T2 weights from Google Drive
├── requirements.txt              # Python dependencies (includes pydicom)
├── .streamlit/
│   └── secrets.toml              # ← YOU CREATE THIS (never commit to git)
└── README.md
```

---

## ⚙️ App Features

| Feature | Description |
|---------|-------------|
| **3-class output** | ET · PD · Healthy with per-class probability bars |
| **T1 / T2 model selection** | Switch between T1-weighted and T2-weighted models from the sidebar — each downloads and caches independently |
| **DICOM support** | Upload `.dcm` files directly; the app converts to PNG using VOI LUT / window-level from the DICOM tags before classifying |
| **DICOM metadata panel** | Displays Patient ID, Modality, Series Description, TE, TR, and slice dimensions after upload |
| **TTA** | Averages original + horizontal-flip predictions (~+1% AUC, +20 ms) |
| **GradCAM** | Class activation map anchored to the Swin `norm` layer |
| **Attention Rollout** | Aggregated activation magnitude across all SwinTransformerBlocks |
| **Brain masking** | Otsu-threshold mask suppresses background noise in heatmaps |
| **Swin-B / Swin-S** | Switchable backbone via sidebar — no code change needed |
| **Auto-download** | Weights fetched from Google Drive at startup, cached in `/tmp` |

---

## 🚀 Deploying to Streamlit Community Cloud

### Step 1 — Host your models on Google Drive

The model checkpoints are too large to commit to GitHub. The app downloads them automatically at startup.

Do this **separately for your T1 and T2 checkpoints**:

1. Upload your `.pt` / `.pth` checkpoint to Google Drive.
2. Right-click → **Share** → **Anyone with the link** → **Viewer** → **Copy link**.
3. Extract the `FILE_ID` from the share URL:
   ```
   https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
                                    ^^^^^^^^^^^^^^^^^^^^
                                    this is your FILE_ID
   ```
4. Repeat for the second checkpoint and keep both IDs handy.

> **Important:** files must be shared as *"Anyone with the link"* — private files will cause the download to fail with a 403.

---

### Step 2 — Push the code to GitHub

```bash
# Create a new repo (private is fine)
git init
git add app.py inference.py model_downloader.py requirements.txt README.md
# Never add secrets.toml — it contains your file IDs
echo ".streamlit/secrets.toml" >> .gitignore
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

### Step 3 — Add secrets on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Connect your GitHub repo, set the main file to `app.py`.
3. Click **Advanced settings → Secrets** and paste:

```toml
GDRIVE_FILE_ID_T1 = "your_t1_file_id_here"
GDRIVE_FILE_ID_T2 = "your_t2_file_id_here"
```

Replace each value with the `FILE_ID` copied in Step 1.

4. Click **Deploy**. On first boot the app will download whichever model the user selects (~300 MB each) into `/tmp/neuroscan_models/` and cache it for the rest of the session.

---

### Step 4 — Configure your `.streamlit/secrets.toml` (local runs)

```toml
# T1-weighted model checkpoint
GDRIVE_FILE_ID_T1 = "your_t1_file_id_here"

# T2-weighted model checkpoint
GDRIVE_FILE_ID_T2 = "your_t2_file_id_here"
```

> **Legacy fallback:** if you only have one model, you can still use `GDRIVE_FILE_ID` (without the `_T1` suffix) and it will map to the T1 slot automatically.

---

### Step 5 — Choose the right backbone variant

| Variant | Parameters | Checkpoint size | Speed (CPU) |
|---------|-----------|-----------------|-------------|
| `swin_b` | 88 M | ~340 MB | ~1.2 s/slice |
| `swin_s` | 49 M | ~195 MB | ~0.7 s/slice |

Select the variant in the sidebar. The variant **must match** the checkpoint you trained — loading a Swin-S checkpoint with the `swin_b` setting (or vice versa) will raise a `RuntimeError`.

---

## 💻 Running Locally

```bash
# 1. Clone / navigate to the project folder
cd neuroscan_app

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the secrets file
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
GDRIVE_FILE_ID_T1 = "your_t1_file_id_here"
GDRIVE_FILE_ID_T2 = "your_t2_file_id_here"
EOF

# 5. Run
streamlit run app.py
```

Models download to `/tmp/neuroscan_models/` on first run and are reused as long as `/tmp` is not cleared.

---

## 📂 Uploading MRI Files

### Standard images (PNG / JPG / TIFF)
Drop any 2D axial slice directly into the upload zone. The image is converted to grayscale internally before classification.

### DICOM files (.dcm)
The app accepts raw DICOM files:

- Window/level (VOI LUT) from the DICOM tags is applied automatically for correct tissue contrast.
- Multi-frame DICOMs are supported — the middle frame is used.
- After upload a metadata panel shows: Patient ID, Modality, Series Description, Rows × Cols, Slice Thickness, TE, and TR.
- The converted PNG is passed to the classifier and XAI pipeline exactly like a normal upload.

> **Dependency:** DICOM support requires `pydicom`, which is already included in `requirements.txt`. If `pydicom` is missing, the app falls back gracefully with a sidebar warning and only accepts image files.

---

## 🏷️ T1 vs T2 Model Selection

Use the **MRI Type** radio button at the top of the sidebar to select the model that matches your scan:

| Type | Tissue contrast | Typical use |
|------|----------------|-------------|
| **T1** | White matter bright, CSF dark | Anatomy, cortical thickness |
| **T2** | CSF bright, lesions hyperintense | Oedema, demyelination, fluid |

Each model has its own Google Drive file ID and cached checkpoint. Switching types downloads the new model only once per session. The active model type is shown as a colour-coded badge (`T1` in teal, `T2` in blue) throughout the UI.

---

## 🔑 Alternative: Skip Google Drive (local checkpoint)

If you already have the `.pt` files on disk, bypass the downloader by editing `model_downloader.py`:

```python
from pathlib import Path

def ensure_model_downloaded(mri_type: str = "T1") -> Path:
    paths = {
        "T1": Path("/path/to/your/swin_t1.pt"),
        "T2": Path("/path/to/your/swin_t2.pt"),
    }
    return paths[mri_type]
```

---

## 🌐 Alternative: Host on Hugging Face Hub

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_HF_USERNAME/neuroscan-mri swin_t1.pt --repo-type model
huggingface-cli upload YOUR_HF_USERNAME/neuroscan-mri swin_t2.pt --repo-type model
```

Then update `model_downloader.py`:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import streamlit as st

FILENAMES = {"T1": "swin_t1.pt", "T2": "swin_t2.pt"}

def ensure_model_downloaded(mri_type: str = "T1") -> Path:
    local = hf_hub_download(
        repo_id="YOUR_HF_USERNAME/neuroscan-mri",
        filename=FILENAMES[mri_type],
        cache_dir="/tmp/neuroscan_models",
        token=st.secrets.get("HF_TOKEN"),
    )
    return Path(local)
```

Add to secrets:
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: Checkpoint not found` | Check `GDRIVE_FILE_ID_T1` / `GDRIVE_FILE_ID_T2` in secrets and that the Drive file is public |
| `RuntimeError: Error(s) in loading state_dict` | Backbone variant mismatch — ensure the sidebar variant matches the uploaded checkpoint |
| `Download failed: 403` | Google Drive file is not shared with "Anyone with the link" |
| `DICOM read failed: ...` | File may be compressed or use a transfer syntax not supported by `pydicom` — try exporting from your PACS as explicit VR little endian |
| `ModuleNotFoundError: pydicom` | Run `pip install pydicom>=2.4.0` or redeploy after adding it to `requirements.txt` |
| `CUDA out of memory` | App runs on CPU by default; no GPU required |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| Slow first load | Normal — each model downloads once (~300 MB). Subsequent loads use the `/tmp` cache. |
| Switching T1↔T2 re-downloads every time | Expected on Streamlit Cloud — `/tmp` is ephemeral. Run locally to persist the cache between sessions. |

---

## ⚠️ Disclaimer

This tool is a **research prototype** developed as part of a thesis project. It is **not** a validated clinical diagnostic device and must **not** be used as a substitute for professional medical evaluation. Always consult a qualified neurologist for diagnosis and treatment decisions.

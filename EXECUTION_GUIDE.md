# MaRS 2026 — Complete Google Colab Execution Guide

---

## Before you start — one-time setup

### Step 1 — Organise your files in Google Drive

Open Google Drive → create this exact folder structure:

```
My Drive/
└── mars_project/
    ├── train/        ← all training images
    ├── val/          ← all validation images
    └── test/         ← all test images
```

**Critical:** every image filename must start with the class name followed by underscore.  
✅ Correct: `sand_001.jpg`, `bedrock_12.jpg`, `soil_7.jpg`  
❌ Wrong: `001.jpg`, `image_sand.jpg`, `IMG_3421.JPG`

If your filenames don't follow this pattern, use the `rename.py` you already have — just update the `base_directory` and `class_name` variables for each class folder.

---

## Phase 1 — Data Augmentation + Transfer Learning

**File:** `PHASE1_Data_and_TransferLearning.py`

**Time needed:** ~30–45 minutes on T4 GPU

### Steps

**1.** Go to [colab.research.google.com](https://colab.research.google.com) → New notebook

**2.** Set GPU runtime:  
`Runtime → Change runtime type → Hardware accelerator → GPU (T4) → Save`

**3.** Copy **CELL 1** into a code cell and run it:
```python
!pip install albumentations -q
```
If it says "restart runtime" → click **Restart** → continue from CELL 2 (don't re-run CELL 1).

**4.** Copy and run each cell in order: CELL 2 → CELL 3 → ... → CELL 12.

**What each cell does:**

| Cell | What happens |
|------|-------------|
| CELL 2 | Mounts Drive, verifies your 3 folders exist |
| CELL 3 | Imports all libraries |
| CELL 4 | Loads images + creates 4× augmented training set |
| CELL 5 | Encodes labels, computes class weights, saves `label_encoder.pkl` |
| CELL 6 | Builds `tf.data` pipeline (batching + prefetch) |
| CELL 7 | Builds EfficientNetB3 model with custom head |
| CELL 8 | **Stage 1 training** — head only, base frozen (~10 epochs) |
| CELL 9 | **Stage 2 fine-tuning** — top 30 layers unfrozen, low LR (~15 epochs) |
| CELL 10 | Evaluates on test set, prints final accuracy |
| CELL 11 | Plots training curves, saves `phase1_training_curves.png` |
| CELL 12 | Saves `mars_phase1_efficientnet.keras` to Drive |

**Expected output after CELL 10:**
```
Test Accuracy: XX.XX%   ← this should be higher than your original model
```

**Files saved to Drive after Phase 1:**
- `mars_phase1_efficientnet.keras`
- `label_encoder.pkl`
- `best_model_checkpoint.keras`
- `phase1_training_curves.png`

---

## Phase 2 — Explainability + Embeddings + Uncertainty

**File:** `PHASE2_Explainability_and_Embeddings.py`

**Time needed:** ~20–30 minutes (MC Dropout takes the longest)

**Prerequisite:** Phase 1 must be complete (model saved to Drive)

### Steps

**1.** Open a **new notebook** in Colab (keep Phase 1 notebook open in another tab if you want).

**2.** Make sure GPU is still enabled: `Runtime → Change runtime type → GPU`

**3.** Run CELL 1 first:
```python
!pip install umap-learn -q
```
Restart runtime if prompted.

**4.** Run cells in order: CELL 2 → ... → CELL 10.

**What each cell does:**

| Cell | What happens |
|------|-------------|
| CELL 2 | Mounts Drive, loads Phase 1 model + label encoder |
| CELL 3 | Loads test images |
| CELL 4 | Auto-detects the last Conv2D layer name for Grad-CAM |
| CELL 5 | Implements Grad-CAM + overlay functions |
| CELL 6 | **Generates Grad-CAM visualizations** for 2 images per class |
| CELL 7 | Extracts 128-dim embedding vectors from all test images |
| CELL 8 | **UMAP + t-SNE** 2D projection plots |
| CELL 9 | **MC Dropout** — 50 passes per image (3–5 min, normal) |
| CELL 10 | Plots uncertainty analysis |

**If CELL 4 prints `None` for the conv layer name:**  
Uncomment this line at the bottom of CELL 4:
```python
last_conv_layer = 'top_conv'
```

**Files saved to Drive after Phase 2:**
- `phase2_gradcam.png`
- `phase2_embeddings.png`
- `phase2_uncertainty.png`
- `embeddings.npy`
- `entropies.npy`

---

## Phase 3 — Few-Shot Learning + Anomaly Detection

**File:** `PHASE3_FewShot_and_AnomalyDetection.py`

**Time needed:** ~10–15 minutes

**Prerequisite:** Phase 1 complete (model saved)

### Steps

**1.** Open a new Colab notebook, enable GPU.

**2.** No new installs needed — seaborn is pre-installed in Colab.

**3.** Run all 12 cells in order.

**What each cell does:**

| Cell | What happens |
|------|-------------|
| CELL 1 | Mounts Drive, loads model + label encoder |
| CELL 2 | Builds 128-dim feature extractor from backbone |
| CELL 3 | Loads train + test images |
| CELL 4 | Extracts feature vectors for all images |
| CELL 5 | Defines prototype functions (compute + predict) |
| CELL 6 | Full-data prototypical classification + classification report |
| CELL 7 | **Few-shot experiment** across K = 1, 3, 5, 10, 20 |
| CELL 8 | Plots few-shot accuracy curve with error bars |
| CELL 9 | Computes anomaly scores (prototype distance method) |
| CELL 10 | Anomaly detection plots |
| CELL 11 | Confusion matrix |
| CELL 12 | Saves `phase3_summary.json` with all results |

**Files saved to Drive after Phase 3:**
- `phase3_few_shot_curve.png`
- `phase3_anomaly.png`
- `phase3_confusion_matrix.png`
- `phase3_summary.json`
- `phase3_results.txt`

---

## Phase 4 — Gradio Interactive Demo

**File:** `PHASE4_Gradio_Demo.py`

**Time needed:** ~5 minutes to launch

**Prerequisite:** Phases 1 + 3 complete

### Steps

**1.** Open a new Colab notebook, enable GPU.

**2.** Run CELL 1:
```python
!pip install gradio -q
```

**3.** Run all 9 cells in order.

**What each cell does:**

| Cell | What happens |
|------|-------------|
| CELL 1 | Installs Gradio |
| CELL 2 | Mounts Drive, loads model, label encoder, phase3 summary |
| CELL 3 | Builds feature extractor |
| CELL 4 | Pre-computes class prototypes from training images |
| CELL 5 | Grad-CAM helper functions (same as Phase 2, inference-ready) |
| CELL 6 | MC Dropout uncertainty function (30 passes, faster) |
| CELL 7 | Single-image anomaly score function |
| CELL 8 | `analyze_mars_image()` — master function called by Gradio |
| CELL 9 | **Launches the Gradio web app** |

**After CELL 9 runs, you will see:**
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

The **public URL** works for anyone in the world for 72 hours.

**What to do with it:**
1. Open the URL → test it with your own images
2. Screenshot the running app
3. Paste URL + screenshot in your GitHub README
4. Add to resume: `Live demo: https://xxxx.gradio.live`

---

## Common errors and fixes

| Error | Fix |
|-------|-----|
| `FileNotFoundError: train/` | Check your Drive folder structure matches exactly `My Drive/mars_project/train/` |
| `KeyError: 'sand'` | A class name in test/val doesn't appear in train — check all image filenames |
| `Could not find layer 'top_conv'` | In Phase 2 CELL 4, uncomment `last_conv_layer = 'top_conv'` |
| `perplexity must be < n_samples` | You have fewer than 30 test images — t-SNE auto-adjusts, just continue |
| GPU not available | `Runtime → Change runtime type → GPU → Save` |
| Gradio URL not appearing | Add `debug=True` inside `demo.launch()` to see the error |
| Runtime disconnects | Colab free tier disconnects after ~1 hour idle. Keep the tab active. |
| `label_encoder.pkl not found` in Phase 2 | You didn't run CELL 5 of Phase 1 — re-run Phase 1 |

---

## What your Drive should look like after all 4 phases

```
My Drive/mars_project/
├── train/                          ← your original images (unchanged)
├── val/
├── test/
├── mars_phase1_efficientnet.keras  ← main trained model
├── best_model_checkpoint.keras     ← backup checkpoint
├── label_encoder.pkl               ← class name mapper
├── phase1_training_curves.png      ← accuracy/loss plots
├── phase2_gradcam.png              ← Grad-CAM visualizations
├── phase2_embeddings.png           ← UMAP + t-SNE plots
├── phase2_uncertainty.png          ← MC Dropout analysis
├── embeddings.npy                  ← raw 128-dim vectors
├── entropies.npy                   ← per-image entropy values
├── phase3_few_shot_curve.png       ← K-shot accuracy plot
├── phase3_anomaly.png              ← anomaly detection plots
├── phase3_confusion_matrix.png     ← confusion matrix
└── phase3_summary.json             ← results used by Phase 4
```

---

## Resume bullet points (copy-paste ready)

```
• Developed Mars surface sediment classifier using EfficientNetB3 transfer learning
  with domain-specific data augmentation (albumentations), achieving [X]% accuracy
  — an improvement over custom CNN baseline built for IRC 2025 (17th place, international).

• Implemented Grad-CAM explainability to visualize which geological features
  (grain size, ripple patterns, rock edges) drive terrain classification decisions.

• Extended to few-shot learning using prototypical networks, enabling classification
  of new terrain types from as few as 5 examples without retraining.

• Integrated Monte Carlo Dropout uncertainty estimation and prototype-distance
  anomaly detection to flag out-of-distribution terrain for human review.

• Deployed as interactive Gradio web application with live public demo.

Tech: Python, TensorFlow/Keras, EfficientNetB3, Albumentations, UMAP, Gradio, Google Colab
```

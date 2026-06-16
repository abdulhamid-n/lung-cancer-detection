# PULMONET — Lung Cancer Detection from Chest CT

A deep-learning project that trains a convolutional neural network to detect lung-cancer
subtypes from chest-CT slices, documents the entire process in a Jupyter notebook, and
presents the results on an interactive website with a live in-browser model demo.

**Authors:** Azizbek Oqbutayev · Nozima Sotiboldiyeva
**Built & documented with:** Claude Code (Opus 4.8)

> ⚠️ **Educational demonstration only — not a medical device.** Trained on a small public
> dataset; must never be used for diagnosis.

---

## Results (held-out test set)

| Metric | Value |
|---|---|
| 4-class accuracy | **67.5%** |
| Macro F1 | 0.70 |
| Macro AUC (one-vs-rest) | 0.87 |
| **Cancer-vs-normal sensitivity** | **1.00** (0 / 241 cancers missed) |
| **Cancer-vs-normal specificity** | **0.98** |

The model is **excellent at the screening question** (is there cancer at all?) and more
modest at distinguishing the three cancer *subtypes* — which overlap heavily on a single
2-D slice. Normal-tissue F1 = 0.99.

## Dataset

**Chest CT-Scan images** — real CT slices, 4 classes (adenocarcinoma, large-cell carcinoma,
squamous-cell carcinoma, normal), pre-split train/valid/test. Hugging Face mirror
[`dorsar/lung-cancer`](https://huggingface.co/datasets/dorsar/lung-cancer) (MIT License);
originally Kaggle (`mohamedhanyyy`). See `docs/` for why this — and not a mythical "MIT
dataset" — was chosen.

## Model

MobileNetV2 pretrained on ImageNet, backbone frozen, with a fresh 4-class head trained by
transfer learning (Adam + cross-entropy, 30 epochs, Apple-MPS). Exported to ONNX (8.9 MB),
verified to match PyTorch to 1.65e-05, and run client-side in the browser via ONNX Runtime Web.

## Project layout

```
lung-cancer-detection/
├── docs/        design spec
├── data/        Chest CT dataset (downloaded via `hf`)
├── notebooks/   lung_cancer_detection.ipynb  ← the full documented pipeline
│                build_notebook.py            ← regenerates the notebook
├── models/      best_model.pt, model.onnx
└── website/     index.html · style.css · app.js · model.onnx + JSON/figures
    └── serve.sh
```

## Reproduce

```bash
# 1. environment (one-time)
conda create -y -n lungai python=3.11
conda run -n lungai pip install numpy pandas scikit-learn matplotlib seaborn pillow \
    jupyter nbconvert nbformat ipykernel torch torchvision onnx onnxruntime onnxscript

# 2. get the dataset (~767 images)
hf download dorsar/lung-cancer --repo-type dataset --local-dir data/chest-ct

# 3. build + run the notebook end-to-end (trains, evaluates, exports ONNX, writes site assets)
conda run -n lungai python notebooks/build_notebook.py
conda run -n lungai jupyter nbconvert --to notebook --execute --inplace \
    notebooks/lung_cancer_detection.ipynb
```

## View the website

The visual report works by opening `website/index.html` directly, but the **live demo**
needs a local server (browsers block loading the ONNX model from `file://`):

```bash
cd website && ./serve.sh      # then open http://localhost:8000
```

Drop in a CT slice (or click a sample) and the model runs entirely in your browser — no
upload, no server-side inference, $0.

## References

- Sandler et al., *MobileNetV2*, CVPR 2018 · Selvaraju et al., *Grad-CAM*, ICCV 2017
- For serious work: LIDC-IDRI · LUNA16 · NLST · NSCLC-Radiomics (NIH-NCI / TCIA)

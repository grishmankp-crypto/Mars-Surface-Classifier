# MaRS 2025*: Mars Surface Sediment Classifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/grishmank/Mars-Surface-Classifier)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)](#)

> **IRC & ERC Competition Model **
> An end-to-end deep learning pipeline for classifying Martian surface terrain, featuring real-time uncertainty estimation and anomaly detection.

**[Click here for web application on Hugging Face Spaces](https://huggingface.co/spaces/grishmank/Mars-Surface-Classifier)**

## Core Pipeline Architecture
This system is designed defensively to handle edge cases for autonomous Martian exploration:

* **Phase 1: Robust Classification:** Fine-tuned `EfficientNetB3` backbone trained to categorize complex Martian terrain types (Sedimentary, Sand, Gravel, Cracked).
* **Phase 2: Uncertainty Estimation:** Implemented **Monte Carlo (MC) Dropout** layers to calculate Shannon Entropy, measuring model confusion and flagging ambiguous frames.
* **Phase 3: Anomaly Detection:** Engineered a **Prototypical Network** distance-metric framework to calculate proximity to known clusters, automatically identifying out-of-distribution (OOD) terrain features.
* **Phase 4 & Deployment:** Built an interactive, serverless UI using **Gradio** and deployed it permanently to the cloud.

## Repository File Structure
* `phase1_training.py` — CNN training & transfer learning pipeline.
* `phase2_uncertainty.py` — MC Dropout implementation and Shannon Entropy calculations.
* `phase3_anomaly.py` — Distance-metric generation for out-of-distribution tracking.
* `app.py` — Main Gradio application script for cloud deployment.
* `requirements.txt` — System dependencies.

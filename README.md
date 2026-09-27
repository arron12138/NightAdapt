<h2 align="center">OnlineStaging: A Multimodal Graph-Temporal Framework for Sleep Stage Classification</h2>

<p align="center">
  This repository contains the model computation code for a personalized online sleep staging project, covering Stage 1 feature extraction, Stage 2 graph-based temporal classification, and Stage 3 personalized online adaptation.
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Sleep%20Staging-green" alt="Sleep staging">
  <img src="https://img.shields.io/badge/Mode-Offline%20%7C%20Online-lightgrey" alt="Offline and online">
</p>

## Introduction

OnlineStaging is a model-side implementation of a sleep stage classification framework designed for multimodal biosignals such as EEG, EOG, and EMG. The repository contains reusable model definitions and online adaptive logic.

The framework contains three major stages:

- Stage 1: Riemannian/SPD feature extraction with MAtt-DA-style cross-dataset feature alignment.
- Stage 2: multimodal graph fusion and causal temporal modeling for sleep stage classification.
- Stage 3: Individualized pseudo-online adaptation based on a frozen reference, with the option to incorporate a prior for sleep stage transitions.

The default sleep-stage label order is:

```text
W, N1, N2, N3, REM
```

## Highlights

### SPD/MAtt-DA Feature Extraction

The Stage1 model maps raw 30-second EEG/EOG/EMG epochs into modality-specific Riemannian features. It uses temporal convolution, covariance construction, SPD projection, matrix rectification, tangent-space mapping, and attention-based feature refinement.

### Dynamic Multimodal Graph Fusion

The Stage2 model treats EEG, EOG, and EMG as modality nodes. Each modality is first encoded into a shared latent space, then fused through dynamic graph message passing or alternative ablation modes such as static graph fusion and feature concatenation.

### Online-Compatible Causal Temporal Modeling

Temporal modeling is performed on fused epoch-level features. The causal TCN setting predicts the current epoch using only current and previous epochs, making it suitable for pseudo-online and real-time sleep staging scenarios.

### Teacher-Student Pseudo-Online Adaptation

The Stage3 online adapter maintains a stable teacher model and an adaptive student model. High-confidence pseudo-labels are selected by confidence, margin, entropy, and teacher-student agreement gates. An optional transition prior can discourage implausible sleep-stage jumps.

## Repository Structure

```text
OnlineStagingGithub/
  models/
    __init__.py
    spd.py
    stage1_matt_da.py
    stage1_matt_da_legacy.py
    stage2_graph.py
    stage2_flexible_fusion.py
    stage2_temporal.py
    online_adapter.py
    losses.py
  MODEL_CODE_MAP.md
  MODEL_CODE_MAP_CN.md
  requirements.txt
  README.md
  README_CN.md
```

## Getting Started

### Environment

Install the full project dependencies with:

```bash
pip install -r requirements.txt
```

The dependency file covers the full Python workflow used by the project: model inference/training, EDF reading and preprocessing, metrics, plotting, progress bars, and spreadsheet export. The core model files themselves only require PyTorch and NumPy, but the full project also uses SciPy, scikit-learn, pandas, matplotlib, MNE, openpyxl, tqdm, and Pillow.

## Data Preparation

 In a typical workflow, data are prepared as 30-second epoch-level features:

Stage1 raw-epoch input:

```text
EEG public channels: [batch_size, eeg_public_channels, samples]
EEG extra channels:  [batch_size, eeg_extra_channels, samples]
EOG channels:        [batch_size, eog_channels, samples]
EMG channels:        [batch_size, emg_channels, samples]
```

Stage2 feature input:

```text
EEG feature: [num_epochs, eeg_dim]
EOG feature: [num_epochs, eog_dim]
EMG feature: [num_epochs, emg_dim]
Label:       [num_epochs]
```

For temporal models, consecutive epochs are grouped into windows:

```text
EEG sequence: [batch_size, seq_len, eeg_dim]
EOG sequence: [batch_size, seq_len, eog_dim]
EMG sequence: [batch_size, seq_len, emg_dim]
```

For causal online inference, the current epoch should be placed at the last position of each sequence window.

## Acknowledgments

This implementation builds on common deep learning components from PyTorch and standard sleep-staging research practices, including multimodal biosignal representation learning, Riemannian feature extraction, temporal context modeling, and online model adaptation.

# AI-Assisted ECG Analysis Tool

## Overview
End-to-end ECG classification system using deep learning.

- Input: ECG signal (.npy, shape: 1000 x 12)
- Output: Multi-label classification (MI, HYP, CD, STTC, NORM)

## Tech Stack
- PyTorch (model)
- FastAPI (backend API)
- Streamlit (frontend UI)
- PTB-XL dataset

## Features
- ECG classification (multi-label)
- Probability-based predictions
- Interactive visualization (Lead I)
- REST API for inference

## Project Structure

```
ecg-classification-system/
│
├── api/
│   └── api.py              # FastAPI backend (model inference)
│
├── scripts/
│   └── predict.py          # Local prediction script
│
├── model/                  # (not included in repo)
│   └── ecg_full_model.pth
│
├── data/                   # (not included in repo)
│   └── sample.npy
│
├── app.py                  # Streamlit frontend UI
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files
```

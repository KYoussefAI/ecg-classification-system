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
│   └── api.py            
│
├── scripts/
│   └── predict.py         
│
├── model/               
│   └── ecg_full_model.pth
│
├── data/                 
│   └── sample.npy
│
├── app.py                 
├── requirements.txt     
├── README.md             
└── .gitignore           
```

## Demo

![ECG App Screenshot](docs/demo.png)

## Pipeline

1. Load ECG signal (.npy)
2. Preprocess signal (reshape to (12,1000))
3. Model inference (CNN)
4. Apply thresholds
5. Return structured diagnosis
6. Display results + visualization

## Model Architecture

- 1D CNN with residual blocks
- Input: (12 leads, 1000 timesteps)
- Output: 5-class multi-label classification
- Loss: BCEWithLogitsLoss

## Performance

- ROC-AUC: ~0.92
- Macro F1-score: ~0.75

## Data

- Dataset: PTB-XL
- Format: WFDB → NumPy
- Shape: (1000, 12)
- Multi-label classification

## Quick Start

uvicorn api.api:app --reload  
streamlit run app.py

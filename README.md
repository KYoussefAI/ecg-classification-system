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

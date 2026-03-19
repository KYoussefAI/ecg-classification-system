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
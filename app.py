import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import io

# ---- CONFIG ----
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="ECG Analysis", layout="centered")

# ---- HEADER ----
st.warning("This tool is for research purposes only. Not for clinical use.")
st.title("AI-Assisted ECG Analysis Tool")
st.write("Upload an ECG file (.npy) for analysis")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("Upload ECG signal", type=["npy"])

if uploaded_file:

    # ---- READ FILE ONCE (CRITICAL) ----
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    if len(file_bytes) == 0:
        st.error("Uploaded file is empty or corrupted")
        st.stop()

    # ---- CALL API ----
    with st.spinner("Analyzing ECG..."):
        response = requests.post(API_URL, files={"file": file_bytes})

    # ---- HANDLE RESPONSE ----
    if response.status_code != 200:
        st.error("Error communicating with API")
        st.stop()

    data = response.json()

    # ---- DIAGNOSIS ----
    st.subheader("Diagnosis")
    st.markdown(f"### {data['diagnosis']}")

    confidence = data.get("confidence", 0)
    st.write(f"Confidence score: {confidence:.2f}")
    # ---- DETAILS ----
    if data.get("details"):
        st.subheader("Detected Conditions")

        for item in data["details"]:
            st.write(f"- {item['condition']} (confidence {item['confidence']:.2f})")
    else:
        st.success("No abnormalities detected")

    # ---- ECG VISUALIZATION ----
    st.subheader("ECG Signal (Lead I)")

    signal = np.load(io.BytesIO(file_bytes))

    fig, axes = plt.subplots(3, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flatten()):
        ax.plot(signal[:, i])
        ax.set_title(f"Lead {i+1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # ---- EXTRA INFO ----
    st.caption("Model trained on PTB-XL dataset (multi-label ECG classification)")
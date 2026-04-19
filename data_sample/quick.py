import numpy as np
import json

x = np.load(r"C:\\Users\\Me\\Desktop\\END TO END DATA ENGINEERING PROJECTS\\ECG\\data_sample\\sample.npy")

with open("sample.json", "w") as f:
    json.dump({
        "signal_data": x.tolist(),
        "patient_name": "test",
        "age": 25,
        "sex": "M"
    }, f)
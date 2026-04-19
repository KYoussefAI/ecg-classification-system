import json
import numpy as np
import os
import random

OUTPUT_DIR = "test_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_ecg_signal():
    """
    Generate a pseudo ECG-like signal (not medically accurate,
    but better than pure random noise).
    Shape: (1000, 12)
    """
    t = np.linspace(0, 1, 1000)

    signal = []

    for lead in range(12):
        base_wave = np.sin(2 * np.pi * 5 * t)  # base rhythm
        noise = np.random.normal(0, 0.1, size=1000)
        variation = np.sin(2 * np.pi * random.uniform(1, 10) * t)

        lead_signal = base_wave + 0.5 * variation + noise
        signal.append(lead_signal)

    signal = np.array(signal).T  # shape (1000, 12)

    return signal.tolist()


def generate_patient_info(i):
    return {
        "patient_name": f"patient_{i}",
        "age": random.randint(20, 80),
        "sex": random.choice(["M", "F"])
    }


def generate_json_file(i):
    data = {
        "signal_data": generate_ecg_signal(),
        **generate_patient_info(i)
    }

    filename = os.path.join(OUTPUT_DIR, f"test_{i}.json")

    with open(filename, "w") as f:
        json.dump(data, f)

    print(f"Created {filename}")


if __name__ == "__main__":
    for i in range(10):
        generate_json_file(i)
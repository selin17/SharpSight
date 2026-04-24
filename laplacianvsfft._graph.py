import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CSV OKU
# -----------------------------
df = pd.read_csv("results.csv")

# -----------------------------
# SIGMA'YA GÖRE ORTALAMA
# -----------------------------
grouped = df.groupby("Sigma").mean(numeric_only=True)

sigmas = grouped.index

laplacian = grouped["Laplacian"]
fft = grouped["FFT"]

# -----------------------------
# NORMALIZATION (Min-Max Scaling)
# -----------------------------
lap_norm = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
fft_norm = (fft - fft.min()) / (fft.max() - fft.min())

# -----------------------------
# PLOT
# -----------------------------
plt.figure()

plt.plot(sigmas, lap_norm, marker='o', label="Laplacian (Normalized)")
plt.plot(sigmas, fft_norm, marker='s', label="FFT (Normalized)")

plt.title("Blur Sensitivity Comparison: Laplacian vs FFT")
plt.xlabel("Sigma (Gaussian Blur Level)")
plt.ylabel("Normalized Response (0-1)")
plt.grid()
plt.legend()

plt.show()
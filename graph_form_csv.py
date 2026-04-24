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

print("\n ORTALAMA DEĞERLER:")
print(grouped)

# -----------------------------
# LAPLACIAN GRAPH
# -----------------------------
plt.figure()
plt.plot(sigmas, grouped["Laplacian"], marker='o')
plt.title("Laplacian Sharpness vs Sigma")
plt.xlabel("Sigma")
plt.ylabel("Sharpness")
plt.grid()
plt.show()

# -----------------------------
# FFT GRAPH
# -----------------------------
plt.figure()
plt.plot(sigmas, grouped["FFT"], marker='o')
plt.title("FFT High Frequency vs Sigma")
plt.xlabel("Sigma")
plt.ylabel("High Frequency Energy")
plt.grid()
plt.show()

# -----------------------------
# PSNR GRAPH
# -----------------------------
plt.figure()
plt.plot(sigmas, grouped["PSNR"], marker='o')
plt.title("PSNR vs Sigma")
plt.xlabel("Sigma")
plt.ylabel("PSNR (dB)")
plt.grid()
plt.show()
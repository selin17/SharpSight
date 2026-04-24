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

print("\nORTALAMA DEĞERLER:")
print(grouped)
# -----------------------------
# PSNR GRAPH (NaN temizlenmiş)
# -----------------------------
psnr_clean = grouped["PSNR"].dropna()

plt.figure()
plt.plot(psnr_clean.index, psnr_clean.values, marker='o')
plt.title("PSNR vs Sigma")
plt.xlabel("Sigma (Gaussian Blur Level)")
plt.ylabel("PSNR (dB)")
plt.grid()
plt.show()
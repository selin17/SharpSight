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
# MSE GRAPH
# -----------------------------
plt.figure()
plt.plot(sigmas, grouped["MSE"], marker='o')
plt.title("MSE vs Sigma")
plt.xlabel("Sigma (Gaussian Blur Level)")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.show()


import cv2
import os
import numpy as np

# -----------------------------
# BASE PATH (OTOMATİK)
# -----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

print("📌 BASE PATH:", base_path)

# -----------------------------
# KLASÖRLER
# -----------------------------
folders = {
    "Original": os.path.join(base_path, "original"),
    "Sigma 2": os.path.join(base_path, "blurred_sigma2"),
    "Sigma 4": os.path.join(base_path, "blurred_sigma4"),
    "Sigma 6": os.path.join(base_path, "blurred_sigma6"),
    "Sigma 8": os.path.join(base_path, "blurred_sigma8"),
}

# -----------------------------
# DEBUG: DOSYA KONTROLÜ
# -----------------------------
for name, folder in folders.items():
    print(f"\n🔎 {name}")
    print("Path:", folder)
    print("Exists:", os.path.exists(folder))

    if os.path.exists(folder):
        files = os.listdir(folder)
        print("File count:", len(files))
        print("Sample files:", files[:5])

# -----------------------------
# LAPLACIAN KERNEL
# -----------------------------
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# -----------------------------
# MANUEL CONVOLUTION
# -----------------------------
def manual_laplacian(image):
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            region = image[i-1:i+2, j-1:j+2]

            value = 0

            for ki in range(3):
                for kj in range(3):
                    value += region[ki, kj] * kernel[ki, kj]

            output[i, j] = value

    return output

# -----------------------------
# SHARPNESS SCORE
# -----------------------------
def sharpness_score(image):
    lap = manual_laplacian(image)
    return np.var(lap)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    results = {}

    for name, folder in folders.items():

        if not os.path.exists(folder):
            print(f"❌ KLASÖR YOK: {folder}")
            continue

        files = os.listdir(folder)

        if len(files) == 0:
            print(f"⚠️ BOŞ KLASÖR: {name}")
            continue

        scores = []
        processed = 0

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(folder, file)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print("⚠️ okunamadı:", file)
                    continue

                score = sharpness_score(image)
                scores.append(score)
                processed += 1

        if processed == 0:
            print(f"⚠️ HİÇ RESİM İŞLENMEDİ: {name}")
            continue

        avg = np.mean(scores)
        results[name] = avg

        print(f"\n✅ {name}")
        print(f"Processed images: {processed}")
        print(f"Average Sharpness: {avg:.2f}")

    # -----------------------------
    # FINAL SONUÇ
    # -----------------------------
    print("\n====================")
    print("FINAL RESULTS")
    print("====================")

    if len(results) == 0:
        print("❌ HİÇ SONUÇ ÜRETİLMEDİ - dataset kontrol et")
    else:
        for k, v in results.items():
            print(f"{k}: {v:.2f}")
import cv2
import os
import numpy as np

# -----------------------------
# HIGH FREQUENCY FFT SHARPNESS
# -----------------------------
def fft_sharpness(image):

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cx, cy = h // 2, w // 2

    # center = low frequency mask
    radius = 20  # adjustable parameter

    mask = np.ones((h, w), dtype=np.float32)
    mask[cx - radius:cx + radius,
         cy - radius:cy + radius] = 0

    # high frequency extraction
    high_freq = magnitude * mask

    # sharpness score (energy of high frequencies)
    score = np.sum(high_freq)

    return score


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("*FFT HIGH FREQUENCY ANALYSIS STARTED*\n")

    base_path = os.path.dirname(os.path.abspath(__file__))

    folders = {
        "Original": os.path.join(base_path, "original"),
        "Sigma 2": os.path.join(base_path, "blurred_sigma2"),
        "Sigma 4": os.path.join(base_path, "blurred_sigma4"),
        "Sigma 6": os.path.join(base_path, "blurred_sigma6"),
        "Sigma 8": os.path.join(base_path, "blurred_sigma8"),
    }

    results = {}

    total_images = 0

    # -----------------------------
    # PROCESS EACH FOLDER
    # -----------------------------
    for name, folder in folders.items():

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {name}")
            continue

        files = os.listdir(folder)

        scores = []
        processed = 0

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):

                path = os.path.join(folder, file)

                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                # optional resize for speed
                image = cv2.resize(image, (128, 128))

                score = fft_sharpness(image)

                scores.append(score)

                processed += 1
                total_images += 1

        if processed > 0:
            avg_score = np.mean(scores)
            results[name] = avg_score

            print(f"{name} -> Processed: {processed} | High-Freq Score: {avg_score:.2f}")
        else:
            print(f"{name} -> No images processed")


    # -----------------------------
    # FINAL RESULTS
    # -----------------------------
    print("\n======================")
    print("FINAL HIGH FREQUENCY RESULTS")
    print("======================")

    print("Total images processed:", total_images)

    for k, v in results.items():
        print(f"{k}: {v:.2f}")
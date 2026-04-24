import cv2
import os
import numpy as np
import csv

# -----------------------------
# PATH
# -----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

original_folder = os.path.join(base_path, "original")

blurred_folders = {
    0: original_folder,
    2: os.path.join(base_path, "blurred_sigma2"),
    4: os.path.join(base_path, "blurred_sigma4"),
    6: os.path.join(base_path, "blurred_sigma6"),
    8: os.path.join(base_path, "blurred_sigma8"),
}

# -----------------------------
# LAPLACIAN KERNEL
# -----------------------------
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# -----------------------------
# SIMPLE LAPLACIAN
# -----------------------------
def laplacian(image):
    h, w = image.shape
    result = np.zeros((h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            value = 0
            for x in range(3):
                for y in range(3):
                    value += image[i + x - 1, j + y - 1] * kernel[x, y]

            result[i, j] = value

    return np.var(result)

# -----------------------------
# SIMPLE FFT
# -----------------------------
def fft_score(image):
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)

    magnitude = np.abs(f_shift)

    return np.mean(magnitude)

# -----------------------------
# MSE
# -----------------------------
def mse(original, blurred):
    return np.mean((original - blurred) ** 2)

# -----------------------------
# PSNR
# -----------------------------
def psnr(mse_val):
    if mse_val == 0:
        return 100
    return 10 * np.log10((255 ** 2) / mse_val)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
results = []

original_files = os.listdir(original_folder)

for file in original_files:

    if not file.endswith((".jpg", ".png", ".jpeg")):
        continue

    original_path = os.path.join(original_folder, file)
    original = cv2.imread(original_path, 0)

    if original is None:
        continue

    original = cv2.resize(original, (256, 256))

    for sigma, folder in blurred_folders.items():

        blur_path = os.path.join(folder, file)

        if not os.path.exists(blur_path):
            continue

        image = cv2.imread(blur_path, 0)

        if image is None:
            continue

        image = cv2.resize(image, (256, 256))

        # -----------------------------
        # CALCULATIONS
        # -----------------------------
        lap = laplacian(image)
        fft = fft_score(image)

        if sigma == 0:
            error = 0
            psnr_val = None
        else:
            error = mse(original, image)
            psnr_val = psnr(error)

        results.append([file, sigma, lap, fft, error, psnr_val])

        print(file, "sigma=", sigma, "done")

# -----------------------------
# SAVE CSV
# -----------------------------
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Sigma", "Laplacian", "FFT", "MSE", "PSNR"])
    writer.writerows(results)

print("CSV saved")
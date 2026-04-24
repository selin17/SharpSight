import cv2
import os
import numpy as np

# -----------------------------
# MSE FUNCTION
# -----------------------------
def mse(original, blurred):
    return np.mean((original - blurred) ** 2)


# -----------------------------
# PSNR FUNCTION
# -----------------------------
def psnr(mse_value):
    if mse_value == 0:
        return 100
    return 10 * np.log10((255 ** 2) / mse_value)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))

    original_folder = os.path.join(base_path, "original")

    blurred_folders = {
        "Sigma 2": os.path.join(base_path, "blurred_sigma2"),
        "Sigma 4": os.path.join(base_path, "blurred_sigma4"),
        "Sigma 6": os.path.join(base_path, "blurred_sigma6"),
        "Sigma 8": os.path.join(base_path, "blurred_sigma8"),
    }

    results = {}

    print("MSE / PSNR ANALYSIS STARTED\n")

    for name, folder in blurred_folders.items():

        mses = []
        psnrs = []

        files = os.listdir(folder)

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                orig_path = os.path.join(original_folder, file)
                blur_path = os.path.join(folder, file)

                original = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
                blurred = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE)

                if original is None or blurred is None:
                    continue

                # resize safety
                original = cv2.resize(original, (256, 256))
                blurred = cv2.resize(blurred, (256, 256))

                error = mse(original, blurred)
                mses.append(error)

                psnrs.append(psnr(error))

        avg_mse = np.mean(mses)
        avg_psnr = np.mean(psnrs)

        results[name] = (avg_mse, avg_psnr)

        print(f"{name} -> MSE: {avg_mse:.2f} | PSNR: {avg_psnr:.2f}")


    print("\n======================")
    print("FINAL RESULTS")
    print("======================")

    for k, v in results.items():
        print(f"{k} -> MSE: {v[0]:.2f} | PSNR: {v[1]:.2f}")
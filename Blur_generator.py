import cv2
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, "original")

sigmas = [2, 4, 6, 8]

for sigma in sigmas:
    os.makedirs(f"blurred_sigma{sigma}", exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(input_folder, filename)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        #  KRİTİK KONTROL
        if image is None:
            print("HATA: açılamadı ->", img_path)
            continue

        for sigma in sigmas:

            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

            output_path = os.path.join(f"blurred_sigma{sigma}", filename)
            cv2.imwrite(output_path, blurred)

print("Blur işlemi tamamlandı.")
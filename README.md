# Automated Blur Detection and Sharpness Assessment Using Spatial and Frequency Domain Analysis

## Overview
This project detects image blur and evaluates image sharpness using spatial and frequency domain analysis techniques.

The main goal is to compare different blur levels and measure image quality through objective metrics.

## Features
- Blur detection using Laplacian variance
- Frequency domain analysis using FFT
- Image sharpness assessment
- Performance evaluation using MSE and PSNR
- Blur simulation with Gaussian blur

## Methods Used

### Spatial Domain Analysis
Laplacian operator is used to detect edge information. Lower variance indicates blurrier images.

### Frequency Domain Analysis
Fast Fourier Transform (FFT) is used to analyze high-frequency components in images.

## Dataset
The dataset used in this project was created manually using original photographs captured by me. 

To simulate different blur levels, artificial blur was applied using Gaussian Blur with sigma values of 2, 4, 6, and 8. This allowed us to analyze the impact of increasing blur intensity on image sharpness and quality.

The original images were used as reference images, while the blurred versions were evaluated using spatial and frequency domain analysis techniques.




## Evaluation Metrics
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Laplacian Variance

## Results
The experiments show that blur increases MSE values and decreases PSNR and Laplacian variance, indicating loss of sharpness.

## Future Work
- Real-time blur detection
- Deep learning based blur classification
- Larger dataset testing


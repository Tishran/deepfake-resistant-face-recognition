
# Deepfake-Resistant Face Recognition Model

This project is a solution for the **Kryptonite ML Challenge 2025**, where the task was to develop a robust face recognition model resistant to deepfake attacks. The model aims to enhance biometric security by accurately distinguishing between real and fake facial images without relying on traditional anti-spoofing modules.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

Deepfake technology poses significant security challenges, particularly in biometric systems like facial recognition. This project addresses these challenges by developing a machine learning model capable of:

1. Detecting deepfake-generated images.
2. Comparing real images of the same person with high accuracy.
3. Differentiating between images of different individuals.

The solution is designed to operate effectively without using external spoofing detection modules, ensuring robustness and scalability.

---

## Features

- **Deepfake Detection**: Achieves high accuracy in identifying manipulated images.
- **Face Matching**: Accurately matches real images of the same person.
- **Scalability**: Optimized for integration into existing biometric systems.
- **Low Error Rates**:
    - Equal Error Rate (EER) $ \approx $ 0.045
- **Real-Time Performance**: Processes inputs with millisecond-level latency.

---

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-repo/deepfake-resistant-face-recognition.git
cd deepfake-resistant-face-recognition
```

2. Run ```setup_venv.sh``` to install fitting python version and all dependencies:

```bash
./scripts/setup_venv.sh
```

3. Run ```download_data.sh``` to download and unpack datasets:
```bash
./scripts/download_data.sh
```


---

## Usage

To use the model for inference:

1. Prepare your input image(s) and csv files and place them in the `data/` directory.
2. Run the inference script:

```bash
python inference.py
```

3. Results will be saved in the `results/` directory.

For training from scratch, refer to `train.py` and `config.py`.

---

## Dataset

The model was trained on a dataset given by organizers of competition.

Preprocessing steps included resizing images to $224 \times 224$, data augmentation, and balancing real vs fake samples.

---

## Model Architecture

The solution leverages a hybrid architecture combining:

1. **Resnet 34**: For feature extraction from facial images.
2. **Triple Loss**: To improve face matching capabilities.

The architecture ensures high performance while maintaining low computational overhead.

---

## Results

### Key Metrics:

- EER: 0.045 on test dataset

---

## Future Work

Potential improvements include:

1. Decreasing EER.
2. Extending support to video-based deepfake detection.

---

## Acknowledgments

This project was developed as part of the Kryptonite ML Challenge organized by IT company "Kryptonite." Special thanks to their AI lab experts for guidance and feedback during the competition.

Contributors:

- Khamatyarov Rushan

---

Feel free to contribute or raise issues in this repository!
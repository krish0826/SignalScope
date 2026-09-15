# SignalScope

### Real vs AI-Generated Image Detection

SignalScope is a computer-vision system designed to estimate whether a single image is **REAL** or **AI-GENERATED**.

It uses transfer learning with **EfficientNet-B0** and is designed with a particular focus on **generalization to AI generators that were not seen during training**.

> **Important:** SignalScope produces a probabilistic estimate, not definitive proof of an image's origin.

---

## 1. Problem Statement

Generative AI can produce increasingly realistic images, making it difficult to distinguish synthetic images from authentic photographs.

SignalScope addresses this challenge by:

* Classifying individual images as REAL or AI-GENERATED
* Providing a confidence score
* Testing performance on an unseen generator
* Supporting real-world photographs
* Providing an explainability prototype using Grad-CAM
* Providing a lightweight web interface for demonstration

---

## 2. Key Features

### Core Detection

* Single-image classification
* REAL / AI-GENERATED prediction
* Confidence score
* Probability breakdown
* GPU-accelerated inference when CUDA is available

### Explainability

SignalScope includes a **Grad-CAM explainability prototype** to visualize image regions that influence the model's prediction.

The goal is to make predictions more interpretable rather than presenting the classifier as a black box.

### Web Interface

A minimal Streamlit interface allows users to:

1. Upload an image
2. Preview the image
3. Analyze the image
4. View the prediction
5. View confidence and probability scores

---

## 3. System Architecture

```text
                 Input Image
                      │
                      ▼
             Image Preprocessing
             Resize → 224×224
                      │
                      ▼
              EfficientNet-B0
             Transfer Learning
                      │
                      ▼
                 Classifier
                 ┌────┴────┐
                 ▼         ▼
               FAKE      REAL
                 │         │
                 └────┬────┘
                      ▼
              Softmax Probabilities
                      │
                      ▼
              Prediction + Confidence
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Streamlit UI       Grad-CAM
```

---

## 4. Model

**Architecture:** EfficientNet-B0

**Training approach:** Transfer learning + fine-tuning

**Input size:** 224 × 224

**Classes:**

```text
0 → FAKE / AI-GENERATED
1 → REAL
```

The model was initialized using ImageNet-pretrained EfficientNet-B0 weights and subsequently fine-tuned on the SignalScope training dataset.

Training used data augmentation techniques including:

* Random resized cropping
* Horizontal flipping
* Small rotations
* Color augmentation
* Gaussian blur
* Random erasing
* ImageNet normalization

These augmentations were intended to improve robustness to changes in image appearance.

---

## 5. Dataset

SignalScope uses multiple data sources for training and evaluation.

### CIFAKE

CIFAKE provides REAL and AI-generated synthetic images.

The original training data was divided into:

* Training: 90,000 images
* Validation: 10,000 images
* Official test: 20,000 images

The official CIFAKE test set was kept separate from training.

Source:

* CIFAKE dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

---

### Custom Real-World Images

A collection of genuine photographs captured using a mobile phone was used to improve the model's exposure to real-world imagery.

Valid images:

```text
Total       : 767
Training    : 613
Validation  : 76
Test        : 78
```

The test portion was kept separate from training.

---

### GenImage

GenImage was used to introduce synthetic images from multiple generators.

Training synthetic images were selected from:

| Generator            |    Images |
| -------------------- | --------: |
| BigGAN               |     5,000 |
| GLIDE                |     1,592 |
| ADM                  |       855 |
| Wukong               |       641 |
| Stable Diffusion 1.5 |       549 |
| VQDM                 |       294 |
| **Total**            | **8,931** |

To specifically test unseen-generator generalization:

**Stable Diffusion 1.4 was not used for training.**

**Midjourney was not used for training.**

Stable Diffusion 1.4 was therefore used as an unseen-generator evaluation set.

GenImage sources:

* Official repository: https://github.com/GenImage-Dataset/GenImage
* GenImage Arrow dataset mirror: https://huggingface.co/datasets/nebula/GenImage-arrow

---

## 6. Final Training Dataset

The V2 training dataset contains:

```text
FAKE  : 53,931
REAL  : 45,613
----------------
TOTAL : 99,544
```

Validation:

```text
FAKE  : 5,000
REAL  : 5,076
----------------
TOTAL : 10,076
```

The final model was trained using the V2 dataset.

---

## 7. Evaluation Results

### CIFAKE Held-Out Test

The final V2 model was evaluated on the official CIFAKE test set.

| Metric              |     Result |
| ------------------- | ---------: |
| Accuracy            | **96.86%** |
| Macro-F1            | **96.85%** |
| ROC-AUC             | **99.78%** |
| False Positive Rate |  **0.61%** |

Confusion matrix:

```text
                 Predicted
              FAKE      REAL

Actual FAKE   9939       61
Actual REAL    568      9432
```

These results represent performance on the CIFAKE held-out test set and should not be interpreted as unseen-generator performance.

---

## 8. Unseen Generator Evaluation

One of the main challenges of AI-generated-image detection is generalization to generators that were not present during training.

SignalScope was evaluated on **Stable Diffusion 1.4**, which was intentionally excluded from training.

```text
Images       : 12,000
Accuracy     : 52.43%
Macro-F1     : 38.93%
ROC-AUC      : 71.01%
```

Confusion matrix:

```text
                 Predicted
              FAKE      REAL

Actual FAKE    325     5675
Actual REAL     33     5967
```

This result demonstrates an important limitation of the current detector: strong performance on the training distribution does not automatically guarantee strong generalization to an unseen generator.

This finding is especially relevant to the SignalScope problem because unseen-generator robustness is a central challenge of synthetic-image detection.

---

## 9. Real-World Test

The final model was also evaluated on 70 genuine photographs captured using a mobile phone.

```text
Total images       : 70
Predicted REAL     : 70
Predicted AI       : 0

REAL prediction rate: 100%
False-positive rate : 0%
```

This is a small real-world sanity test and should not be considered a statistically representative benchmark.

---

## 10. Custom AI Sanity Test

A small custom test was performed using 10 synthetic images from the GLIDE generator.

```text
AI images tested : 10
Correctly detected: 10
Result           : 10/10
```

This is a small sanity check rather than a formal benchmark.

---

## 11. Explainability

SignalScope includes a Grad-CAM prototype for visual explanation.

Grad-CAM can highlight image regions that contribute strongly to the model's prediction.

This is intended to support:

* Model interpretability
* Visual inspection
* Debugging
* Understanding possible model failure modes

The explanation should not be interpreted as proof that a specific region was created by AI.

---

## 12. Web Application

SignalScope uses **Streamlit** for its demonstration interface.

Run:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\app.py
```

The application provides:

* Image upload
* Image preview
* REAL / AI-GENERATED classification
* Confidence score
* Probability breakdown
* Minimal user interface

---

## 13. Project Structure

```text
SignalScope/
│
├── app/
│   └── app.py
│
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── custom_real/
│   ├── custom_real_split/
│   ├── genimage_fake/
│   └── combined_v2/
│
├── models/
│   └── signalscope_efficientnet_b0_v2_final.pth
│
├── src/
│   ├── train.py
│   ├── train_v2.py
│   ├── resume_v2.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── evaluate_unseen_local.py
│   ├── test_real_world.py
│   └── gradcam_test.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 14. Installation

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SignalScope
```

### Create virtual environment

```powershell
python -m venv venv
```

### Install dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 15. Run Prediction

```powershell
.\venv\Scripts\python.exe src\predict.py
```

For the web application:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\app.py
```

---

## 16. Hardware

SignalScope was developed and tested using an NVIDIA GPU with CUDA acceleration.

GPU acceleration is automatically used when CUDA is available.

CPU inference is also supported, although training and inference may be slower.

---

## 17. Limitations

SignalScope has several important limitations:

* AI-generated image detection is probabilistic.
* Performance can decrease on unseen generators.
* Image compression, resizing, screenshots, and editing can affect predictions.
* The current unseen-generator evaluation demonstrates that high in-domain accuracy does not guarantee cross-generator robustness.
* The real-world phone-photo evaluation is relatively small.
* Confidence values are model scores and should not be interpreted as guaranteed probabilities.
* The current system focuses on image-level classification rather than identifying a specific generator.

Therefore, SignalScope should be used as a **decision-support tool**, not as definitive evidence about image authenticity.

---

## 18. Responsible Use

SignalScope is intended for research and educational use in detecting synthetic imagery.

Predictions should be described as:

> **"Likely AI-generated"**

rather than definitive accusations.

The system should not be used as the sole basis for decisions involving individuals, identity, political claims, or other high-impact situations.

---

## 19. Reproducibility

The project includes:

* Training code
* Evaluation code
* Inference code
* Model weights
* Dataset preparation scripts
* Streamlit application
* Grad-CAM prototype
* Dependency specification

The official CIFAKE test set was kept separate from training.

Stable Diffusion 1.4 and Midjourney were intentionally excluded from training for unseen-generator evaluation.

---

## 20. Originality Declaration

This project was developed as part of the SIH 2026 internal hackathon development period.

The implementation uses publicly available datasets, open-source machine-learning libraries, and an ImageNet-pretrained EfficientNet-B0 backbone where permitted by the problem statement.

No public end-to-end project or notebook was directly submitted as the solution.

AI-assisted development tools were used for development assistance, debugging, and documentation, while the project-specific dataset preparation, experimentation, evaluation, model selection, and integration were performed as part of this project.

---

## 21. Future Improvements

Potential future improvements include:

* Better unseen-generator generalization
* Additional generator diversity
* Frequency-domain artifact analysis
* Robustness testing under JPEG compression and screenshots
* Probability calibration
* Generator attribution
* C2PA / EXIF provenance analysis
* Image-text consistency checks
* More faithful localized explanations
* Larger real-world evaluation datasets

---

## 22. Acknowledgements & References

### CIFAKE

Bird, J. J. and Lotfi, A., *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.*

Dataset:

https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

### GenImage

GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.

Official repository:

https://github.com/GenImage-Dataset/GenImage

Dataset mirror used for selected Arrow shards:

https://huggingface.co/datasets/nebula/GenImage-arrow

### Frameworks

* PyTorch
* TorchVision
* Streamlit
* scikit-learn
* Grad-CAM

---

## 23. Demo

Local demo:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\app.py
```

Demo video / deployed application:

**Add link before final submission.**

---

# SignalScope

**Detecting synthetic media with computer vision — while being honest about uncertainty.**

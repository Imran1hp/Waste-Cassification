# Waste vs Not-Waste Image Classification

A lightweight binary image classification system that classifies an image as **Waste** or **Not Waste** using **MobileNetV3-Small**. The model is designed for deployment on resource-constrained edge devices such as a **Raspberry Pi**.

## Project Overview

The goal of this project is to develop a compact and efficient image classifier that can distinguish between:

- **Waste**
- **Not Waste**

The final selected model is **MobileNetV3-Small with ImageNet pretrained weights**, fine-tuned for the two-class classification problem.

The project workflow is:

```text
Image
  ↓
Resize to 224 × 224
  ↓
MobileNetV3-Small
  ↓
Softmax
  ↓
Waste / Not Waste
```

## Objectives

- Build a binary waste image classifier.
- Use a lightweight CNN suitable for Raspberry Pi.
- Apply transfer learning with ImageNet pretrained weights.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Test the classifier on real-world images.
- Prepare the model for Raspberry Pi deployment.

## Model

The project uses **MobileNetV3-Small** from Torchvision.

```python
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = torch.nn.Linear(1024, 2)
```

The final classifier predicts two classes:

```text
0 → Not Waste
1 → Waste
```

### Why MobileNetV3-Small?

MobileNetV3-Small was selected because the target deployment platform is a Raspberry Pi.

Advantages:

- Lightweight architecture
- Low computational requirements
- Small memory footprint
- Fast inference
- Suitable for edge devices
- Supports ImageNet transfer learning

## Transfer Learning

The selected model uses ImageNet pretrained weights:

```python
mobilenet_v3_small(weights="DEFAULT")
```

Instead of learning all visual features from random initialization, the model starts with features learned from ImageNet and adapts them to the Waste/Not Waste problem.

The original classifier was replaced with:

```python
torch.nn.Linear(1024, 2)
```

This allows the model to produce predictions for the two target classes.

## Dataset

The project uses a binary image dataset containing:

```text
Waste
Not Waste
```

The selected Model_4 experiment used approximately:

- **20,000 training images**
- **3,000 test images**

The dataset contains waste and non-waste images from real-world visual conditions.

## Image Preprocessing

Images were resized to:

```text
256 × 256 pixels
```

The model uses ImageNet normalization:


Training augmentation included horizontal flipping:

```python
transforms.RandomHorizontalFlip(p=0.5)
```

Images were resized before training to reduce the CPU image-loading and preprocessing bottleneck observed during the initial training experiments.

## Training

Training was performed using **PyTorch** on **Google Colab** with an NVIDIA Tesla T4 GPU.

### Main Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV3-Small |
| Weights | ImageNet pretrained |
| Task | Binary classification |
| Classes | 2 |
| Input size | 256 × 256 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | CrossEntropyLoss |
| Batch size | 64 |
| Training images | ~20,000 |
| Test images | ~3,000 |
| GPU | NVIDIA Tesla T4 |

Loss function:

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

Optimizer:

```python
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.0001
)
```

## Model_4 Results

Model_4 uses:

```python
mobilenet_v3_small(weights="DEFAULT")
```

The recorded training results were:

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.0922 | 97% | 0.1992 | 94% |
| 1 | 0.0187 | 99% | **0.1899** | **95%** |
| 2 | 0.0099 | 100% | 0.2954 | 94% |
| 3 | 0.0062 | 100% | 0.2912 | 95% |
| 4 | 0.0051 | 100% | 0.2871 | 95% |

The best recorded test accuracy was approximately:

```text
95%
```

The lowest recorded test loss was:

```text
0.1899
```

at Epoch 1.

## Overfitting Analysis

The model learned the training data very quickly.

For example:

```text
Epoch 1
Train Accuracy = 99%
Test Accuracy  = 95%
```

Afterward, training loss continued to decrease while test loss increased:

```text
Train Loss:
0.0187 → 0.0099 → 0.0062 → 0.0051

Test Loss:
0.1899 → 0.2954 → 0.2912 → 0.2871
```

This indicates the beginning of overfitting.

Because the model reached high performance very quickly, unnecessary additional epochs were avoided.

## Model Experiments

Several MobileNetV3-Small experiments were performed.

### Model_3 — Training From Scratch

Model_3 used:

```python
mobilenet_v3_small(weights=None)
```

Approximate result:

```text
Train Accuracy: ~99%
Test Accuracy:  ~87%
```

Although the model learned the training data very well, its generalization performance was substantially lower.

### Model_4 — Transfer Learning

Model_4 used:

```python
mobilenet_v3_small(weights="DEFAULT")
```

Approximate result:

```text
Train Accuracy: 100%
Test Accuracy:  95%
```

Comparison:

| Model | Initialization | Best Test Accuracy |
|---|---|---:|
| Model_3 | Random / None | ~87% |
| **Model_4** | **ImageNet pretrained** | **~95%** |

This experiment demonstrated that transfer learning provided a significant improvement for this classification task.

## Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Not Waste", "Waste"]
    )
)
```

Predictions can be collected with:

```python
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_dataloader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
```

## Real-World Testing

A separate folder containing approximately **100 real-life images** was also used for qualitative testing.

These images were unlabeled, so accuracy and F1-score cannot be calculated directly from them. Instead, the model predictions and confidence scores can be inspected manually.

Example:

```text
Image: image001.jpg
Prediction: Waste
Confidence: 98.42%

Image: image002.jpg
Prediction: Not Waste
Confidence: 96.17%
```

Real-world testing is important because high test-set accuracy does not necessarily guarantee strong performance under different lighting, backgrounds, camera angles, and environments.

## Raspberry Pi Deployment

The main deployment target is a Raspberry Pi.

The intended inference pipeline is:

```text
Raspberry Pi Camera
       ↓
Capture Image
       ↓
Resize to 224 × 224
       ↓
MobileNetV3-Small
       ↓
Prediction
       ↓
Waste / Not Waste
```

The trained PyTorch model can later be converted to an edge-friendly format such as:

- ONNX
- TorchScript
- TensorFlow Lite

The final format can be selected based on the Raspberry Pi hardware and required inference speed.

## Training Performance Optimization

During the initial experiments, training was slower than expected even though the model and tensors were on the GPU.

Profiling showed that GPU transfer and model computation were relatively fast, while image loading and preprocessing created a significant bottleneck.

The original images were around:

```text
1365 × 1024
```

while the network input was:

```text
224 × 224
```

Pre-resizing the dataset reduced unnecessary image-processing work during training and improved the training pipeline.

## Project Structure

A suggested repository structure is:

```text
waste-classification/
│
├── dataset/
│   ├── Waste/
│   └── Not Waste/
│
├── models/
│   └── model_4/
│       └── best_model.pth
│
├── notebooks/
│   └── training.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── predictions.csv
│
├── test_images/
│
├── requirements.txt
│
└── README.md
```

## Installation

Install the required Python packages:

```bash
pip install torch torchvision scikit-learn matplotlib pandas pillow
```

## Loading the Model

```python
import torch
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(1024, 2)

model.load_state_dict(
    torch.load("models/model_4/best_model.pth")
)

model.eval()
```

Class names:

```python
class_names = [
    "Not Waste",
    "Waste"
]
```

## Image Prediction

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0)
```

Prediction:

```python
with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

print("Prediction:", class_names[prediction.item()])
print("Confidence:", confidence.item() * 100)
```

## Key Findings

### Transfer learning was effective

Using:

```python
weights="DEFAULT"
```

performed substantially better than training MobileNetV3-Small from scratch with:

```python
weights=None
```

### MobileNetV3-Small is appropriate for edge deployment

The architecture provides a useful balance between:

- Accuracy
- Model size
- Computational requirements
- Inference speed

This makes it a suitable candidate for Raspberry Pi deployment.

### Early stopping is important

The model reached high training accuracy very quickly. Continuing training after validation/test performance stopped improving increased the risk of overfitting.

### Real-world testing is necessary

The final system should be tested with images from environments and conditions different from the training dataset.

## Future Improvements

- Create a dedicated validation set.
- Use validation data rather than the test set for model selection.
- Add early stopping.
- Perform systematic hyperparameter tuning.
- Evaluate the final model on a fixed, untouched test set.
- Analyze the confusion matrix.
- Test additional real-world images.
- Quantize the model for edge deployment.
- Convert the model to ONNX, TorchScript, or TFLite.
- Benchmark inference time on Raspberry Pi.
- Connect the classifier to a Raspberry Pi camera.
- Build a real-time waste classification application.

## Final Selected Model

```text
Model:          MobileNetV3-Small
Initialization: ImageNet pretrained weights
Task:           Binary classification
Classes:        Waste / Not Waste
Input Size:     224 × 224
Framework:      PyTorch
Optimizer:      Adam
Learning Rate:  0.001
Batch Size:     32
Target Device:  Raspberry Pi
```

**Selected model: Model_4**

Model_4 was selected because it combines a lightweight MobileNetV3-Small architecture with ImageNet transfer learning and achieved approximately **95% test accuracy** in the recorded experiment.

## Author

**Ali Imran**

Waste vs Not-Waste Classification using MobileNetV3-Small

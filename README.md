# Brain Tumor Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-red)
![Project](https://img.shields.io/badge/Type-DeepLearning-green)

This project was built as a learning exercise in deep learning and model interpretability.

## Installation and Setup

Clone the project and go to the project directory

```bash
git clone https://github.com/spyridouladev/brain-tumor-classifier.git

cd brain-tumor-classifier
```

Create virtual environment

```bash
python -m venv venv
```

Activate it
```bash
source venv/bin/activate # Linux / Mac
```

```bash
venv\Scripts\activate # Windows
```

Install dependencies
```bash
pip install -r requirements.txt
```

## GPU Setup and Execution Options

- If you want to use GPU acceleration, install the correct PyTorch version for your hardware.

- If you are using CPU only, you can skip directly to the [Training and Evaluation Pipeline](#training-and-evaluation-pipeline) section.

- If you are just testing the pretrained model, you can skip to the [Interactive Demo Application](#interactive-demo-application) section.

# AMD 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```
# NVIDIA
 ```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Training and Evaluation Pipeline

# preprocess dataset
```bash
python preprocessing.py
```

# train model
```bash
python train.py
```

# evaluate model
```bash
python test.py
```

## Interactive Demo Application
```bash
python app.py
```

## Dataset

This project uses the Brain Tumor MRI dataset from Kaggle:

https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset

The dataset contains ~10,000 MRI images across 4 classes:
- glioma
- meningioma
- pituitary
- no tumor
# Brain Tumor Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-red)
![Project](https://img.shields.io/badge/Type-DeepLearning-green)

This project was built as a learning exercise in deep learning and model interpretability.

## Run Locally

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
## Run

```bash
# preprocess dataset
python preprocessing.py

# train model
python train.py

# evaluate model
python test.py

# generate Grad-CAM visualization
python gradcam.py
```

Gradcam outputs are saved in the `results/` directory.

## Dataset

This project uses the Brain Tumor MRI dataset from Kaggle:

https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset

The dataset contains ~10,000 MRI images across 4 classes:
- glioma
- meningioma
- pituitary
- no tumor
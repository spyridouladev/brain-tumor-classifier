import gradio as gr
import torch

from gradcam import GradCAM, run_gradcam
from libs.model_loader import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(device)
gradcam_obj = GradCAM(model)

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

def predict(image, aug):
    overlay, display, pred = run_gradcam(
        model,
        gradcam_obj,
        image,
        device,
        class_names,
        aug
    )
    return overlay, pred

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(["None", "Blur", "Black box (center)", "Noise"])
    ],
    outputs=[
        gr.Image(label="Grad-CAM"),
        gr.Text(label="Prediction")
    ],
    title="Brain Tumor Grad-CAM"
)

demo.launch()
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

from libs.my_transforms import get_eval_transform, get_display_transform
from libs.utils import get_base_path
from libs.model_loader import load_model

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        backbone = self.model.model
        target_layer = backbone.layer4
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        output = self.model(x)
        pred_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, pred_class].backward()

        grads = self.gradients[0]
        activations = self.activations[0]

        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=x.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.detach().cpu(), pred_class.item()

def overlay_heatmap(cam, image):
    image = np.array(image).astype(np.uint8)

    cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

    return overlay

def choose_augmentation():
    print("\nChoose augmentation:")
    print("1 - None")
    print("2 - Blur")
    print("3 - Black box (center)")
    print("4 - Noise")

    return input("Enter choice: ")

def apply_augmentation(image, choice):
    if choice == "2":
        image = image.filter(ImageFilter.GaussianBlur(radius=5))
    elif choice == "3":
        img_np = np.array(image)
        h, w, _ = img_np.shape
        img_np[h//3:2*h//3, w//3:2*w//3] = 0
        image = Image.fromarray(img_np)
    elif choice == "4":
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_np)

    return image

def preprocess_image(img_path, choice):
    eval_transform = get_eval_transform()
    display_transform = get_display_transform()

    image = Image.open(img_path).convert("RGB")
    image = apply_augmentation(image, choice)

    input_tensor = eval_transform(image).unsqueeze(0)
    display_image = display_transform(image)

    return input_tensor, display_image

def run_gradcam(model, gradcam_obj, image, device, class_names, augmentation="None"):
    image = apply_augmentation(image, augmentation)

    eval_transform = get_eval_transform()
    display_transform = get_display_transform()

    input_tensor = eval_transform(image).unsqueeze(0).to(device)
    display_image = display_transform(image)

    cam, pred = gradcam_obj.generate(input_tensor)

    overlay = overlay_heatmap(cam, display_image)

    return overlay, display_image, class_names[pred]

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.processing.interfaces.main import IExplainer


class SimpleGradCam(IExplainer):
    """
    GradCAM
    """

    def __init__(self):
        self.activations = None

    def hook_fn(self, module, input, output):
        self.activations = output

    def _find_last_conv_layer(self, model):
        """
        find last convolutional layer
        """
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def generate_heatmap(self, image: np.ndarray, model: any, target_layer=None) -> np.ndarray:
        """
        Generate a HeatMap ensuring we use a spatial layer.
        """
        # 1. Prepare image to PyTorch (HWC -> CHW, Normalize)
        img_tensor = torch.from_numpy(image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # 2. Identify the last convolutional layer safely
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
            if target_layer is None:
                raise ValueError("No se encontró una capa Conv2d en el modelo para generar el Heatmap.")

        # 3. Registry Hook
        handle = target_layer.register_forward_hook(self.hook_fn)

        # 4. Forward pass
        with torch.no_grad():
            model(img_tensor)

        handle.remove()

        # 5. Process activation
        # acts shape esperado: [1, Channels, H, W] (ej: 1, 1280, 7, 7)
        acts = self.activations

        if acts is None:
            raise RuntimeError("El hook no capturó activaciones.")

        # Average over channels (Fast-CAM)
        heatmap = torch.mean(acts, dim=1).squeeze()

        # 6. Robust Normalization (Min-Max Scaling)
        heatmap = F.relu(heatmap)

        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)

        if max_val - min_val > 0:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap.cpu().numpy()
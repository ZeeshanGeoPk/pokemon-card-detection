import torch
from torchvision import transforms as VT
from PIL import Image


class CardEmbeddor():
    def __init__(self, model_version='dinov2_vits14', device='cuda'):
        self.model = torch.hub.load('facebookresearch/dinov2', model_version).to(device)
        self.model.eval()
        self.device = device
        
        self.pre_transforms = VT.Compose([
            VT.Resize((224, 224)),
            # VT.CenterCrop(224*2),
            VT.ToTensor(),
            VT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        
    def embed_images(self, images_pil):
        images_pil = [img.convert('RGB') for img in images_pil]
        batch_tensor = self.preprocess_img(images_pil).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch_tensor)
        return embeddings
        
    def preprocess_img(self, images_pil):
        batch_tensor = torch.stack([self.pre_transforms(img) for img in images_pil])
        return batch_tensor
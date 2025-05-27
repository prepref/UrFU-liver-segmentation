
import numpy as np
import pydicom


import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image(dcm_path):
    """Загружает и обрабатывает пару DICOM"""
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept

    # Медицински значимая нормализация
    hu_min, hu_max = -1000, 2000  # Стандартный диапазон для КТ снимков
    img = np.clip(img, hu_min, hu_max)
    img_normalized = (img - hu_min) / (hu_max - hu_min)  # [0, 1]

    return img_normalized

def preprocess_image(image):
    transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)

    return transformed['image'].unsqueeze(0).float()
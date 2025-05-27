import os
import cv2
import numpy as np
from svgwrite import Drawing

def save_prediction_results(image_tensor, pred_mask, output_dir='results'):
    """
    Сохраняет оригинальное изображение и контуры предсказания
    с поддержкой как RGB, так и grayscale изображений
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_np = image_tensor.squeeze().cpu().numpy()
    
    if image_np.ndim == 2 or image_np.shape[0] == 1:
        image_np = (image_np * 255).astype(np.uint8)
        if image_np.ndim == 3:
            image_np = image_np[0]
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    jpg_path = os.path.join(output_dir, 'image.jpg')
    cv2.imwrite(jpg_path, image_np)
    
    contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    svg_path = os.path.join(output_dir, 'contour.svg')
    dwg = Drawing(svg_path, size=(pred_mask.shape[1], pred_mask.shape[0]))
    
    for contour in contours:
        if len(contour) > 2:
            points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
            dwg.add(dwg.polyline(
                points, 
                fill='none', 
                stroke='green', 
                stroke_width=2,
                stroke_linejoin='round'
            ))
    
    dwg.save()
    
    print(f"Оригинальное изображение сохранено: {jpg_path}")
    print(f"Контуры сохранены в SVG: {svg_path}")


   
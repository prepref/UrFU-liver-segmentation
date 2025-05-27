from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import os
import numpy as np
import torch
import logging
import shutil
from torchvision.models.segmentation import deeplabv3_resnet50
from datetime import datetime

from preprocess import load_image, preprocess_image
from postprocess import save_prediction_results

app = FastAPI(title="Service Cancer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Инициализация модели
model = deeplabv3_resnet50(num_classes=1)
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model_path = "./best_model_DeepLabV3.pth"
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()  # Переводим модель в режим оценки

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем директорию для сохранения результатов
RESULTS_DIR = os.path.join('/app/data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.dcm'):
            raise HTTPException(status_code=400, detail="File must be in DICOM format")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_dir = os.path.join(RESULTS_DIR, f'process_{timestamp}')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, 'wb') as f:
            contents = await file.read()
            f.write(contents)
        logger.info(f'File saved to: {temp_path}')

        
        input_dicom = load_image(temp_path)
        processed_image = preprocess_image(input_dicom)
        with torch.no_grad():
            output = model(processed_image)['out']
        pred = torch.sigmoid(output.squeeze(0)) > 0.7
        pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        print(pred_mask.max())

        temp_output_dir = os.path.join(temp_dir, "results")
        os.makedirs(temp_output_dir, exist_ok=True)
        save_prediction_results(processed_image, pred_mask, output_dir=temp_output_dir)
        print(0)
        
        final_dir = "../svg-liver-editor/img" 
        os.makedirs(final_dir, exist_ok=True)
        for fname in ["image.jpg", "contour.svg"]:
            src = os.path.join(temp_output_dir, fname)
            dst = os.path.join(final_dir, f"{fname}")
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")


        shutil.rmtree(temp_dir)

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
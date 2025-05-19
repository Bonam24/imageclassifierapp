import os
from flask import render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Upload folder setup
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Only include required models
MODEL_INFO = {
    "vgg19": ("models/vgg19_model.keras", 224),
    "resnet": ("models/resnet50_model.keras", 224),
    "inception": ("models/inceptionv3_model.keras", 299),
}

def load_selected_models():
    models = {}
    for name, (path, size) in MODEL_INFO.items():
        models[name] = {
            "model": load_model(path),
            "size": size
        }
    return models

def preprocess_image(file_like, target_size):
    img = Image.open(file_like).convert('RGB')
    img = img.resize((target_size, target_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_image(file_like, models):
    results = {}
    for name, info in models.items():
        model = info["model"]
        size = info["size"]
        img = preprocess_image(file_like, size)
        pred = model.predict(img, verbose=0)[0][0]
        results[name] = float(pred)
    avg_pred = np.mean(list(results.values()))
    final_class = "camera" if avg_pred > 0.5 else "screen"
    return {"individual_predictions": results, "final_prediction": final_class}

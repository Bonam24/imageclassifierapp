import os
import base64
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set upload folder and ensure it exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mapping model names to paths and input sizes
MODEL_INFO = {
    "vgg16": ("models/vgg16_model.keras", 224),
    "vgg19": ("models/vgg19_model.keras", 224),
    "resnet": ("models/resnet50_model.keras", 224),
    "inception": ("models/inceptionv3_model.keras", 299),
    "efficientnet": ("models/efficientnetb0_model.keras", 224),
}

def load_all_models():
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

models = load_all_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    
    # Secure the filename to prevent directory traversal
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the file
    file.save(filepath)

    # Reopen the file for prediction
    with open(filepath, 'rb') as img_file:
        prediction = predict_image(img_file, models)

    # Generate URL to display image
    image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', prediction=prediction, image_url=image_url)

from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from utils.predict import load_all_models, predict_image
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
models = load_all_models()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction=None, error="No file uploaded")
    
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction=None, error="No file selected")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    with open(filepath, "rb") as f:
        prediction = predict_image(f, models)

    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

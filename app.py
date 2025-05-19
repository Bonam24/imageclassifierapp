from flask import Flask, request, render_template
from utils.predict import load_all_models, predict_image
import os

app = Flask(__name__)


# Load models once on startup
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
    
    # Get prediction from your models
    prediction = predict_image(file, models)
    
    return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

# //added code 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'healthy_vs_rotten.h5'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Try to load the model, handle if not present
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
class_names = ['Biodegradable', 'Recyclable', 'Trash']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model file not found. Please train and save the model as 'healthy_vs_rotten.h5' in the project directory.", 500
    if 'image' not in request.files:
        return "No image uploaded", 400
    img_file = request.files['image']
    if img_file.filename == '':
        return "No selected file", 400
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class)

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio_details.html')

if __name__ == '__main__':
    app.run(debug=True) 
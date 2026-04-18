import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("gi_model.keras")

# Class labels (IMPORTANT: same order as training)
classes = ['normal', 'polyps', 'ulcer']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    # Prediction
    prediction = model.predict(img)[0]

    probs = [round(p * 100, 2) for p in prediction]

    result = classes[np.argmax(prediction)]
    confidence = max(probs)

    # 🔥 Stage + Remedy logic
    stage = ""
    remedy = ""

    if result == "ulcer":
        if confidence > 80:
            stage = "Severe"
            remedy = "Consult doctor immediately. Avoid spicy food and alcohol."
        elif confidence > 60:
            stage = "Moderate"
            remedy = "Take medication and maintain proper diet."
        else:
            stage = "Mild"
            remedy = "Follow healthy diet and avoid oily food."

    elif result == "polyps":
        stage = "Needs Medical Check"
        remedy = "Consult doctor for further diagnosis."

    else:
        stage = "Normal"
        remedy = "Maintain healthy lifestyle."

    return render_template('index.html',
                           prediction=result,
                           confidence=confidence,
                           stage=stage,
                           remedy=remedy,
                           probs=probs,
                           classes=classes)

if __name__ == "__main__":
    app.run(debug=True)
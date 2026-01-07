import pickle
from flask import Flask, request, app, jsonify,url_for, render_template
import joblib
import numpy as np
import pandas as pd

import os

app = Flask(__name__)

# Use absolute paths for everything
model_path = r"C:\Users\user\OneDrive\Desktop\endtoendproject\wine_cultivar\wine_cultivar\wine_svm_model.joblib"
scaler_path = r"C:\Users\user\OneDrive\Desktop\endtoendproject\wine_cultivar\wine_cultivar\scaler.joblib"

# Load them using the full paths
with open(model_path, 'rb') as f:
    model = joblib.load(f)

with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Collect all 13 features from the form
        input_data = [
            float(request.form.get('alcohol', 0)),
            float(request.form.get('malic_acid', 0)),
            float(request.form.get('ash', 0)),
            float(request.form.get('alcalinity_of_ash', 0)),
            float(request.form.get('magnesium', 0)),
            float(request.form.get('total_phenols', 0)),
            float(request.form.get('flavanoids', 0)),
            float(request.form.get('nonflavanoid_phenols', 0)),
            float(request.form.get('proanthocyanins', 0)),
            float(request.form.get('color_intensity', 0)),
            float(request.form.get('hue', 0)),
            float(request.form.get('od280_od315', 0)),
            float(request.form.get('proline', 0))
        ]
        
        # 2. Prepare data and scale
        final_input = np.array(input_data).reshape(1, -1)
        new_data = scaler.transform(final_input)
        
        # 3. Predict the class
        prediction = model.predict(new_data)[0]
        
        # 4. Map the numeric result to an attractive name
        names = {
            0: "Premium Cultivar (Class 0)",
            1: "Standard Cultivar (Class 1)",
            2: "Economy Cultivar (Class 2)"
        }
        
        final_name = names.get(prediction, f"Unknown Class {prediction}")
        
        # 5. Return to the webpage
        return render_template('home.html', prediction_text=final_name)

    except Exception as e:
        # In case of an error, show the error message on the page
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
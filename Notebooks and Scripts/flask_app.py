from flask import Flask, request, jsonify
import pandas as pd
import os
import numpy as np
import cloudpickle
import sys
import traceback

app = Flask(__name__)



#global variables to store model components
model = None
feature_columns = None
min_max_scaler = None
model_loaded = False

# File paths
model_path = 'accident_prediction_model3.pkl'
features_path = 'model_features3.pkl'
scaler_path = 'min_max_scaler3.pkl'

def load_file(filepath):
    """Helper function to load a file using cloudpickle"""
    try:
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        raise

def load_model():
    global model, feature_columns, min_max_scaler, model_loaded
    try:
        print("Loading model components...")
        model = load_file(model_path)
        feature_columns = load_file(features_path)
        min_max_scaler = load_file(scaler_path)
        model_loaded = True
        print("All files loaded successfully!")
        return True
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        model_loaded = False
        return False

#Initial load attempt
load_model()

@app.route('/reload-model-verbose', methods=['GET'])
def reload_model_verbose():
    global model, feature_columns, min_max_scaler, model_loaded

    results = {
        "files": {
            "model": os.path.exists(model_path),
            "features": os.path.exists(features_path),
            "scaler": os.path.exists(scaler_path)
        },
        "loaded": {},
        "errors": []
    }

    components = {
        "model": model_path,
        "features": features_path,
        "scaler": scaler_path
    }

    for name, path in components.items():
        try:
            with open(path, 'rb') as f:
                obj = cloudpickle.load(f)
                globals()[name] = obj
                results["loaded"][name] = True
        except Exception as e:
            results["loaded"][name] = False
            results["errors"].append({
                "file": name,
                "error": str(e),
                "details": traceback.format_exc()
            })

    model_loaded = all(results["loaded"].values())
    results["all_loaded"] = model_loaded

    return jsonify(results)
#for checking the versions of the packages
@app.route('/versions', methods=['GET'])
def versions():
    import sklearn
    return jsonify({
        "sklearn_version": sklearn.__version__,
        "cloudpickle_version": cloudpickle.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__
    })
#for checking the status of the model
@app.route('/model-status', methods=['GET'])
def model_status():
    return jsonify({
        "files": {
            "model": os.path.exists(model_path),
            "features": os.path.exists(features_path),
            "scaler": os.path.exists(scaler_path)
        },
        "model_loaded": model_loaded
    })
#for reloading the model
@app.route('/reload-model', methods=['GET'])
def reload_model():
    success = load_model()
    return jsonify({"success": success, "model_loaded": model_loaded})

#forr predicting the accident
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        year = data.get('year')
        month = data.get('month')

        if not year or not month:
            return jsonify({"error": "Missing year or month"}), 400

        input_data = pd.DataFrame({'Year': [year], 'Month_Numeric': [month]})[feature_columns]
        prediction = model.predict(input_data)[0]
        prediction = round(min_max_scaler.inverse_transform([[prediction]])[0][0])

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50001)
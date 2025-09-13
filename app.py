from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    with open('new_fertilizer_classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

try:
    with open('labelEncoder_01.pkl', 'rb') as f: 
        label_encoder = pickle.load(f)
    logger.info("Label Encoder  loaded successfully")
except FileNotFoundError:
    logger.error("Label encoder file not found")
    label_encoder = None
except Exception as e:
    logger.error(f"Error loading label label_encoder file: {str(e)}")
    label_encoder = None 

try: 
    with open("model/crop_recommendation.pkl", "rb") as f:
        crop_recommender = joblib.load(f)
    logger.info("Crop recommendation model loaded successfully")
except FileNotFoundError:
    logger.error("Crop recommendation file not found")
    crop_recommender = None
except Exception as e:
    logger.error(f"Error loading crop recommendation model: {str(e)}")
    crop_recommender = None

try:
    with open("model/scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    logger.info("Scaler successfully loaded")
except FileNotFoundError:
    logger.error("Scaler file not found")
    scaler = None
except Exception as e:
    logger.error(f"SOME OTHER ERROR ON SCALER, {e}")
    scaler = None

try: 
    with open("model/yield_prediction_model.pkl", "rb") as f:
        yield_predictor = pickle.load(f)
    logger.info("Yield predictor loaded successfully")
except FileNotFoundError:
    logger.error("Yield prediction file not found")
    yield_predictor = None
except Exception as e:
    logger.error(f"SOME OTHER ERROR ON YIELD PREDICTION, {e}")
    yield_predictor = None

def preprocess_input(data):
    """Preprocess input data for model prediction"""
    try:

        logger.info(f"Received data: {data}")
        for field in ['District_Name', 'Soil_color', 'Crop']:
            if not data.get(field, ''):
                raise ValueError("Missing required categorical fields")
            
        row = {
            "District_Name": data['District_Name'],
            'Soil_color': data['Soil_color'],
            'Crop': data['Crop'],
            "Nitrogen" : float(data.get("Nitrogen", 0)),
            "Phosphorus": float(data.get("Phosphorus", 0)),
            "Potassium"  : float(data.get("Potassium", 0)),
            "pH": float(data.get("pH",  0)),
            "Rainfall": float(data.get("Rainfall", 0)),
            "Temperature": float(data.get("Temperature", 25))
        }

        return pd.DataFrame([row])

    except (ValueError, TypeError) as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Invalid input  data: {str(e)}")
    
@app.route('/')
def index():
    """Serve the main HTML page"""
    districts = {"Kohlapur", "Solapur", "Sangli", "Pune", "Satara"}
    return render_template('index.html', districts=districts)

# Crop metadata
crop_metadata = {
    'rice': {'msp': 2369, 'season': 'Kharif', 'average_yield': 2.218494541018379, 'yield_quintals': 22.1849454101838},
    'maize': {'msp': 2249, 'season': 'Kharif', 'average_yield': 3.4272159774543587, 'yield_quintals': 34.27215977454359},
    'chickpea': {'msp': 5440, 'season': 'Kharif'},
    'kidneybeans': {'msp': 6000, 'season': 'Kharif'},
    'pigeonpeas': {'msp': 8000, 'season': 'Kharif'},
    'mothbeans': {'msp': 5000, 'season': 'Kharif'},
    'mungbean': {'msp': 8768, 'season': 'Kharif'},
    'blackgram': {'msp': 7800, 'season': 'Kharif'},
    'lentil': {'msp': 6025, 'season': 'Kharif'},
    'pomegranate': {'msp': 2500, 'season': 'Kharif'},
    'banana': {'msp': 900, 'season': 'Kharif', 'average_yield': 26.85112785404081, 'yield_quintals': 268.51127854040817},
    'mango': {'msp': 1200, 'season': 'Kharif'},
    'grapes': {'msp': 1800, 'season': 'Kharif'},
    'watermelon': {'msp': 800, 'season': 'Kharif'},
    'muskmelon': {'msp': 700, 'season': 'Kharif'},
    'apple': {'msp': 3500, 'season': 'Kharif'},
    'orange': {'msp': 1600, 'season': 'Kharif'},
    'papaya': {'msp': 600, 'season': 'Kharif'},
    'coconut': {'msp': 1000, 'season': 'Kharif', 'average_yield': 8652.000198744186, 'yield_quintals': 86520.00198744186},
    'cotton': {'msp': 7710, 'season': 'Kharif'},
    'jute': {'msp': 4750, 'season': 'Kharif', 'average_yield': 7.555392696430939, 'yield_quintals': 75.5539269643094},
    'coffee': {'msp': 15000, 'season': 'Kharif'}
}

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    """Recommend the yield that the farmer is going to get"""
    try:
        if yield_predictor is None:
            return jsonify({
                'error': 'Yield predictor model is not available',
                'message': "The model couldn't be loaded"
            }), 500
        
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data provided",
                "message": "Please provide input data for prediction"
            }), 400
        
        yield_inputs = {
            'Nitrogen': float(data.get("Nitrogen", 0)),
            "phosphorus": float(data.get("Phosphorus", 0)),
            "potassium": float(data.get('Potassium', 0)),
            'Temperature_Celsius_rec': float(data.get('Temperature', 0)),
            "humidity": 82.00274,
            "ph": float(data.get('pH', 0)),
            "Rainfall_mm_rec": float(data.get('Rainfall', 0)),
            "Region": "South",
            "Soil_Type": data['Soil_color']
        }

        feature_list = [
            yield_inputs['Nitrogen'],
            yield_inputs['phosphorus'],
            yield_inputs['potassium'],
            yield_inputs['Temperature_Celsius_rec'],
            yield_inputs['humidity'],
            yield_inputs['ph'],
            yield_inputs['Rainfall_mm_rec'],
            yield_inputs['Region'],
            yield_inputs['Soil_Type']
        ]

        features = pd.DataFrame([feature_list], columns=['Nitrogen', 'phosphorus', 'potassium', 'Temperature_Celsius_rec', 'humidity', 'ph', 'Rainfall_mm_rec', 'Region', 'Soil_Type'])

        prediction = yield_predictor.predict(features)[0]

        # Convert numpy type to native Python type for JSON serialization
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        else:
            prediction = float(prediction)

        logger.info(f"YIELD PREDICTION NUMBER {prediction}")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "message": "YIELD PREDICTION MODEL"
        })
    except ValueError as e:
        logger.error(f"Yield validation error: {str(e)}")
        return jsonify({
            'error': "Validation error",
            "message" : str(e)
        }), 400
    except Exception as e:
        logger.error(f"Crop predction error: {str(e)}")
        return jsonify({
            'error': 'Prediction error',
            'message': "An error occured while making the crop predction"
        }), 500
        
@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction for fertilizer recommendation requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model is not available',
                'message': 'The machine learning model could not be loaded'
            }), 500
        
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message' : 'Please provide input data for prediction'
            }), 400
        
        features = preprocess_input(data)

        #make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        #get fertilzier name
        fertilizer_name = label_encoder.inverse_transform([prediction])[0]

        prediction = int(prediction)
        confidence = float(np.max(prediction_proba)) * 100

        logger.info(f"Prediction made: {fertilizer_name} (confidence: {confidence:.2f}%)")

        return jsonify({
            'success': True,
            'fertilizer': str(fertilizer_name),
            'confidence': round(confidence, 2),
            'message': f'Recommended fertilizer: {fertilizer_name}'
        })

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction error',
            'model_loaded': "An error occured while making the prediction"
        }), 500

@app.route('/health', methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status' : 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

@app.route('/predict-crop', methods=['POST'])
def predict_crop_new():
    """Handle crop prediction requests at /predict-crop"""
    try:
        if crop_recommender is None or scaler is None:
            return jsonify({
                'error': 'Crop model is not available',
                'message': 'The crop recommendation model could not be loaded'
            }), 500
        
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide input data for crop prediction'
            }), 400

        # Extract and validate crop prediction inputs
        crop_inputs = {
            'N': float(data.get('Nitrogen', 0)),
            'P': float(data.get('Phosphorus', 0)),
            'K': float(data.get('Potassium', 0)),
            'temperature': float(data.get('Temperature', 0)),
            'humidity': 70.0,  # Default humidity since it's not in the form
            'ph': float(data.get('pH', 0)),
            'rainfall': float(data.get('Rainfall', 0))
        }

        # Create feature list in correct order
        feature_list = [
            crop_inputs['N'],
            crop_inputs['P'],
            crop_inputs['K'],
            crop_inputs['temperature'],
            crop_inputs['humidity'],
            crop_inputs['ph'],
            crop_inputs['rainfall']
        ]

        # Create DataFrame with feature names
        features = pd.DataFrame([feature_list], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Scale the input features
        scaled_features = scaler.transform(features)

        # Get prediction probabilities
        probabilities = crop_recommender.predict_proba(scaled_features)[0]
        
        # Get top 5 predictions
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_crops = crop_recommender.classes_[top5_indices]
        
        recommendations = []
        for crop in top5_crops:
            meta = crop_metadata.get(crop, {})
            profit = meta.get('yield_quintals', 0) * meta.get('msp', 0)
            recommendations.append({
                'crop': crop.capitalize(),
                'profit': f'₹{profit:,.2f}',
                'yield': f"{meta.get('yield_quintals', 'N/A')} quintals",
                'season': meta.get('season', 'N/A'),
            })

        logger.info(f"Crop prediction made: {[r['crop'] for r in recommendations]}")

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'message': 'Crop recommendations generated successfully'
        })

    except ValueError as e:
        logger.error(f"Crop validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Crop prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction error',
            'message': 'An error occurred while making the crop prediction'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
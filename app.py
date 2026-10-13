from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained AI model once when the server starts
model = joblib.load('crop_model.pkl')

# 1. Serve the website when someone visits the main URL
@app.route('/')
def home():
    return render_template('index.html')

# 2. Create the API endpoint to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent by the JavaScript
        data = request.json
        
        # Package the data exactly how the AI expects it (DataFrame with column names)
        input_df = pd.DataFrame([[
            data['N'], data['P'], data['K'], 
            data['temperature'], data['humidity'], data['ph'], data['rainfall']
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Ask the AI for the prediction
        prediction = model.predict(input_df)[0]
        
        # Send the result back to the website
        return jsonify({'success': True, 'crop': prediction})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Make a prediction using the trained model
    prediction = model.predict(input_data)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

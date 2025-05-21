from flask import Flask, render_template, request, jsonify
import logging
import joblib
import numpy as np

model = joblib.load("Notebooks/models/linear_regression_model.pkl")
scaler = joblib.load("Notebooks/models/scaler.pkl")


app = Flask(__name__)

logging.basicConfig(
    filename='flask.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    data = request.get_json()
    try:
        area = float(data['number1'])
        scaled_area = scaler.transform([[area]])
        predicted_price = model.predict(scaled_area)[0]

        return jsonify({
            'status': 'success',
            'price': round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

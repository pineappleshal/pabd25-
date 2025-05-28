from flask import Flask, render_template, request, jsonify
import logging
import joblib
import numpy as np

# Загрузка моделей
model_1room = joblib.load("../notebooks/models/xgboost_model_1room_v1.pkl")
model_2room = joblib.load("../notebooks/models/xgboost_model_2room_v1.pkl")
model_3room = joblib.load("../notebooks/models/xgboost_model_3room_v1.pkl")
model_4room = joblib.load("../notebooks/models/xgboost_model_4room_v1.pkl")

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
        rooms = int(data['number2'])
        floors = int(data['number3'])
        floor = int(data['number4'])

        # Без использования scaler
        input_data = np.array([[area, rooms, floors, floor]])

        # Выбор модели в зависимости от количества комнат
        if rooms == 1:
            model = model_1room
        elif rooms == 2:
            model = model_2room
        elif rooms == 3:
            model = model_3room
        elif rooms == 4:
            model = model_4room
        else:
            return jsonify({'status': 'error', 'message': 'Невозможное количество комнат.'}), 400

        # Предсказание
        predicted_price = float(model.predict(input_data)[0])

        return jsonify({
            'status': 'success',
            'price': round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

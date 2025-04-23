from flask import Flask, render_template, request, jsonify
import logging

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
        area = float(data.get('number1', 0))
        price_per_m2 = 300000
        total_price = area * price_per_m2

        return jsonify({
            'status': 'success',
            'price': round(total_price, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pandas as pd
from src.mlops.pipelines import prediction_service

# Load the CSV file once when the API starts
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))

app = Flask(__name__, template_folder=os.path.join(abs_path, "templates"))

bbunique2u3 = pd.read_csv(os.path.join(abs_path, 'bbunique2u3.csv'))
intro_html = '/data/zhuanghao/MyGithub/MLOps-ForecastSys/assets/templates/intro.html'
@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    company_id = int(data.get('company_id'))
    year_to_predict = data.get('year_to_predict')
    y_to_predict = data.get('y_to_predict')

    if not company_id or not year_to_predict or not y_to_predict:
        return jsonify({'error': 'Missing required parameters'}), 400

    if year_to_predict != 2026:
        return jsonify({'message': 'Prediction only available for 2026.'}), 400
    try:
        company_name = bbunique2u3[bbunique2u3.COMPANY_ID == company_id]['Company_name'].iloc[0]
        id_bb_unique = bbunique2u3[bbunique2u3.COMPANY_ID == company_id]['ID_BB_UNIQUE'].iloc[0]
    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid company_id provided'}), 400

    try:
        prediction = prediction_service("LGBRegression", id_bb_unique, y_to_predict, year_to_predict)
        return jsonify({
            'company_name': company_name,
            'company_id': company_id,
            'year_to_predict': year_to_predict,
            'y_to_predict': y_to_predict,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)

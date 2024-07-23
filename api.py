from flask import Flask, jsonify
import requests
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)

# Define el token de autenticación y la URL base de la API desde las variables de entorno
auth_token = os.getenv('AUTH_TOKEN')
base_url = os.getenv('BASE_URL')

# Categorías
categories = ['Service', 'Driver', 'Traffic Sign', 'Navigation', 'Stops', 'Travel Time', 'Behavior']

# Diccionario para almacenar datos
data_dict = {
    'general': {
        'realData': [],
        'predictData': []
    },
    'categories': []
}

# Función para procesar y analizar los datos
def analyze_data(api_url, category=None):
    headers = {
        'Authorization': f'{auth_token}'
    }

    response = requests.get(api_url, headers=headers)
    data = response.json()

    if not data['data']:
        print(f"No hay suficientes datos para {'la categoría ' + category if category else 'los datos generales'}")
        return

    df = pd.json_normalize(data['data'])
    df['creationDate'] = pd.to_datetime(df['creationDate'])
    df = df[['creationDate', 'score']]
    df.set_index('creationDate', inplace=True)
    time_series = df.resample('MS').mean()

    # Asegúrate de que el rango de fechas sea mensual
    full_range = pd.date_range(start=time_series.index.min(), end=time_series.index.max(), freq='MS')
    time_series = time_series.reindex(full_range)
    time_series.interpolate(method='linear', inplace=True)

    num_observations = len(time_series)
    print(f'Número de observaciones para {"la categoría " + category if category else "los datos generales"}: {num_observations}')

    period = 0
    if num_observations < 4:
        print("Not enough data to predict")
        return
    if 24 > num_observations > 4:
        period = int(num_observations / 2)
    else:
        period = 12

    decompose = seasonal_decompose(time_series, model='additive', period=period)

    model_HW = ExponentialSmoothing(time_series, seasonal_periods=period, trend='add', seasonal='add', initialization_method='estimated').fit()

    # Predicción de los últimos 3 meses y los próximos 3 meses
    start_pred = time_series.index[-2]  # Incluir el último mes de datos reales
    end_pred = time_series.index[-1] + pd.DateOffset(months=3)
    pred_HW = model_HW.predict(start=start_pred, end=end_pred)

    if category is None:
        # Guardar datos generales
        global data_dict
        data_dict['general']['realData'] = [{'date': date.strftime('%B'), 'score': score} for date, score in time_series.reset_index().values]
        data_dict['general']['predictData'] = [{'date': date.strftime('%B'), 'RealScore': None, 'HWScore': score} for date, score in pred_HW.reset_index().values]
    else:
        # Guardar datos por categoría
        category_data = {
            'realData': [{'date': date.strftime('%B'), f'realScore{category}': score} for date, score in time_series.reset_index().values],
            'predictData': [{'date': date.strftime('%B'), f'RealScore{category}': None, f'HWScore{category}': score} for date, score in pred_HW.reset_index().values]
        }
        data_dict['categories'].append(category_data)


# Función para combinar datos de categorías
def combine_category_data():
    combined_data = []
    # Obtén todas las fechas de los datos generales
    general_dates = [entry['date'] for entry in data_dict['general']['realData']]

    for date in general_dates:
        entry = {'date': date}
        for cat_data in data_dict['categories']:
            for real in cat_data['realData']:
                if real['date'] == date:
                    category_name = categories[data_dict['categories'].index(cat_data)]
                    entry.update({
                        f'realScore{category_name}': real[f'realScore{category_name}']
                    })
            for pred in cat_data['predictData']:
                if pred['date'] == date:
                    category_name = categories[data_dict['categories'].index(cat_data)]
                    entry.update({
                        f'HWScore{category_name}': pred[f'HWScore{category_name}']
                    })
        combined_data.append(entry)

    return combined_data

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        # Análisis general
        analyze_data(os.getenv('BASE_URL2'))

        # Análisis por categoría
        for category in categories:
            analyze_data(base_url + category, category)

        # Obtener datos combinados de categorías
        data_dict['categories'] = combine_category_data()

        return jsonify({
            'status': 'success',
            'data': data_dict,
            'message': 'hw succeed'
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'data': [],
            'message': 'not enough data'
        }), 417

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)

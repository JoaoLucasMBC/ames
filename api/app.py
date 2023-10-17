import csv
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict_price():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    
    with open('colunas.txt', 'r') as file:
        columns = [col.strip() for col in file.readlines()]

    data = request.get_json(force=True)
    for col in data.keys():
        if col not in columns:
            return jsonify({"msg": "Column {} does not exist".format(col)}), 400


    data = request.get_json(force=True)
    most_occ = pd.read_csv('most_occuring.csv')

    # Create a dictionary with JSON data and for missing values grab the one from most_occ df
    row_data = {col: data.get(col, most_occ[col].iloc[0]) for col in columns}

    # Create a DataFrame with one row
    df = pd.DataFrame([row_data])

    # save df as csv
    df.to_csv('new_data.csv', index=False)

    row_data_serializable = {k: int(v) if isinstance(v, np.int64) else v for k, v in row_data.items()}
    return jsonify({"msg": "Success", "data": row_data_serializable}), 200




if __name__ == '__main__':
    app.run(debug=True)

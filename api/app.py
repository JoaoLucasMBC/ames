from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from feature_engeneering import feature_engeneering


app = Flask(__name__)


@app.route('/ames/predict', methods=['POST'])
def predict_price():
    if not request.is_json:
        return jsonify({"Mensagem": "Faltando JSON na requisição"}), 400
    
    columns = pd.read_csv("most_occuring.csv").columns.tolist()
        

    data = request.get_json(force=True)
    for col in data.keys():
        if col not in columns:
            return jsonify({"Mensagem": "Coluna {} não existe".format(col)}), 400


    data = request.get_json(force=True)

    data_most = pd.read_csv('/Users/pedropertusi/Desktop/4o semestre/ML/ames/api/most_occuring.csv')

    row_data = {}

    for col in columns:
        if col in data.keys():
            row_data[col] = data[col]
        else:
            row_data[col] = data_most[col][0]

    # Create a DataFrame with one row
    df = pd.DataFrame([row_data])


    # I need the Dtype to be the same as the original df

    continuous_variables = ['Lot.Frontage','Lot.Area','Mas.Vnr.Area','BsmtFin.SF.1','BsmtFin.SF.2','Bsmt.Unf.SF','Total.Bsmt.SF','X1st.Flr.SF','X2nd.Flr.SF','Low.Qual.Fin.SF','Gr.Liv.Area','Garage.Area','Wood.Deck.SF','Open.Porch.SF','Enclosed.Porch','X3Ssn.Porch','Screen.Porch','Pool.Area','Misc.Val']

    discrete_variables = ['Year.Built','Year.Remod.Add','Bsmt.Full.Bath','Bsmt.Half.Bath','Full.Bath','Half.Bath','Bedroom.AbvGr','Kitchen.AbvGr','TotRms.AbvGrd','Fireplaces','Garage.Yr.Blt','Garage.Cars','Mo.Sold','Yr.Sold',] 

    categorical_variables = ['MS.SubClass','MS.Zoning','Street','Alley','Land.Contour','Lot.Config','Neighborhood','Condition.1','Condition.2','Bldg.Type','House.Style','Roof.Style','Roof.Matl','Exterior.1st','Exterior.2nd','Mas.Vnr.Type','Foundation','Heating','Central.Air','Garage.Type','Misc.Feature','Sale.Type','Sale.Condition',] 

    for col in continuous_variables:
        df[col] = df[col].astype('float64')

    for col in categorical_variables:
        df[col] = df[col].astype('category')

    for col in discrete_variables:
        df[col] = df[col].astype('float64')

    category_orderings = {
    'Lot.Shape': ['Reg','IR1','IR2','IR3',],
    'Utilities': ['AllPub','NoSewr','NoSeWa','ELO',],
    'Land.Slope': ['Gtl','Mod','Sev',],
    'Overall.Qual': [1,2,3,4,5,6,7,8,9,10,],
    'Overall.Cond': [1,2,3,4,5,6,7,8,9,10,],
    'Exter.Qual': ['Ex','Gd','TA','Fa','Po',],
    'Exter.Cond': ['Ex','Gd','TA','Fa','Po',],
    'Bsmt.Qual': ['Ex','Gd','TA','Fa','Po',],
    'Bsmt.Cond': ['Ex','Gd','TA','Fa','Po',],
    'Bsmt.Exposure': ['Gd','Av','Mn','No','NA',],
    'BsmtFin.Type.1': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf',],
    'BsmtFin.Type.2': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf',],
    'Heating.QC': ['Ex','Gd','TA','Fa','Po',],
    'Electrical': ['SBrkr','FuseA','FuseF','FuseP','Mix',],
    'Kitchen.Qual': ['Ex','Gd','TA','Fa','Po',],
    'Functional': ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal',],
    'Fireplace.Qu': ['Ex','Gd','TA','Fa','Po',],
    'Garage.Finish': ['Fin','RFn','Unf',],
    'Garage.Qual': ['Ex','Gd','TA','Fa','Po',],
    'Garage.Cond': ['Ex','Gd','TA','Fa','Po',],
    'Paved.Drive': ['Y','P','N',],
    'Pool.QC': ['Ex','Gd','TA','Fa',],
    'Fence': ['GdPrv','MnPrv','GdWo','MnWw',],
}
    
    for col, orderings in category_orderings.items():
        df[col] = df[col].astype('category').cat.set_categories(orderings, ordered=True)



    
    x = feature_engeneering(df)
    return jsonify({"Mensagem": "O valor do imóvel foi predito com sucesso!", "Valor Predito": f"${x:.2f}"}), 200




if __name__ == '__main__':
    app.run(debug=True)

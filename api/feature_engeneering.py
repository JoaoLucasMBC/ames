import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def feature_engeneering(df_novo):
    for col in df_novo.select_dtypes('category'):
        df_novo[col] = df_novo[col].astype(object)
    

    df_novo['MS.SubClass'] = df_novo['MS.SubClass'].apply(lambda x: str(x))
    df_novo['Mo.Sold'] = df_novo['Mo.Sold'].apply(lambda x: str(x))
    df_novo['Yr.Sold'] = df_novo['Yr.Sold'].apply(lambda x: str(x))

    cols_to_change_to_num = ['Lot.Shape', 'Land.Contour', 'Land.Slope', 'Exter.Qual', 'Exter.Cond', 'Bsmt.Qual', 'Bsmt.Cond', 'Bsmt.Exposure', 'Heating.QC', 'Kitchen.Qual', 'Functional', 'Garage.Finish', 'Paved.Drive']

    df_novo['Lot.Shape'].replace({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, inplace=True)
    df_novo['Land.Contour'].replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)
    df_novo['Land.Slope'].replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace=True)
    df_novo['Exter.Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Exter.Cond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Bsmt.Qual'].replace({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Bsmt.Cond'].replace({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Bsmt.Exposure'].replace({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)
    df_novo['Heating.QC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Kitchen.Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df_novo['Functional'].replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)
    df_novo['Garage.Finish'].replace({'NoGarage':0, 'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)
    df_novo['Paved.Drive'].replace({'N':1, 'P':2, 'Y':3}, inplace=True)

    df_novo.loc[(df_novo['Mas.Vnr.Type'] == 'None') & (df_novo['Mas.Vnr.Area'] > 1), 'Mas.Vnr.Type'] = 'BrkFace' 
    df_novo.loc[(df_novo['Mas.Vnr.Type'] == 'None') & (df_novo['Mas.Vnr.Area'] == 1), 'Mas.Vnr.Area'] = 0 
    for vnr_type in df_novo['Mas.Vnr.Type'].unique():
        df_novo.loc[(df_novo['Mas.Vnr.Type'] == vnr_type) & (df_novo['Mas.Vnr.Area'] == 0), 'Mas.Vnr.Area'] = \
        df_novo[df_novo['Mas.Vnr.Type'] == vnr_type]['Mas.Vnr.Area'].mean() 

    idx_to_drop = df_novo[df_novo['Gr.Liv.Area'] >= 4000].index
    df_novo.drop(idx_to_drop, inplace=True)

    # Total Square Footage
    df_novo['Total.SF'] = df_novo['Total.Bsmt.SF'] + df_novo['Gr.Liv.Area']
    df_novo['Total.Floor.SF'] = df_novo['X1st.Flr.SF'] + df_novo['X2nd.Flr.SF']
    df_novo['Total.Porch.SF'] = df_novo['Open.Porch.SF'] + df_novo['Enclosed.Porch'] + \
    df_novo['X3Ssn.Porch'] + df_novo['Screen.Porch']

    # Total Bathrooms
    df_novo['Total.Bathrooms'] = df_novo['Full.Bath'] + .5 * X['Half.Bath'] + \
    df_novo['Bsmt.Full.Bath'] + .5 * df_novo['Bsmt.Half.Bath']

    # Booleans
    df_novo['Has.Basement'] = df_novo['Total.Bsmt.SF'].apply(lambda x: 1 if x > 0 else 0)
    df_novo['Has.Garage'] = df_novo['Garage.Area'].apply(lambda x: 1 if x > 0 else 0)
    df_novo['Has.Porch'] = df_novo['Total.Porch.SF'].apply(lambda x: 1 if x > 0 else 0)
    df_novo['Has.Pool'] = df_novo['Pool.Area'].apply(lambda x: 1 if x > 0 else 0)
    df_novo['Was.Completed'] = (df_novo['Sale.Condition'] != 'Partial').astype(np.int64)

    boolean_features = ['Has.Basement', 'Has.Garage', 'Has.Porch', 'Has.Pool', 'Was.Completed']

    numeric_cols = df_novo.select_dtypes(include=[np.number]).columns
    numeric_cols = [f for f in numeric_cols if f not in boolean_features]

    cat_cols = df_novo.select_dtypes(include=['object']).columns

    skew_limit = 0.5
    skew_vals = df_novo[numeric_cols].skew()

    high_skew = skew_vals[abs(skew_vals) > skew_limit]
    skew_cols = high_skew.index.tolist()

    for col in skew_cols:
        df_novo[col] = np.log1p(df_novo[col])
        df_novo[col] = df_novo[col].astype(np.float64)
    
    #get dummies do df_novo, normaliza usando o standarscalizer do sklearn e retorna o df_novo
    df_novo = pd.get_dummies(df_novo, drop_first=True)


    scaler = StandardScaler()
    df_novo.loc[:,numeric_cols] = scaler.fit_transform(df_novo[numeric_cols])
    df_novo.loc[:,numeric_cols] = scaler.transform(df_novo[numeric_cols])
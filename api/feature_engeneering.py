import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_SEED = 42

def feature_engeneering(df_novo):

    X = df_novo.copy()
    for col in X.select_dtypes('category'):
        X[col] = X[col].astype(object)

    X['MS.SubClass'] = X['MS.SubClass'].apply(lambda x: str(x))
    X['Mo.Sold'] = X['Mo.Sold'].apply(lambda x: str(x))
    X['Yr.Sold'] = X['Yr.Sold'].apply(lambda x: str(x))



    X['Lot.Shape'].replace({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, inplace=True)

    X['Land.Contour'].replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace=True)

    X['Land.Slope'].replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace=True)

    X['Exter.Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Exter.Cond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Bsmt.Qual'].replace({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Bsmt.Cond'].replace({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Bsmt.Exposure'].replace({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)

    X['Heating.QC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Kitchen.Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)

    X['Functional'].replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)

    X['Garage.Finish'].replace({'NoGarage':0, 'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)

    X['Paved.Drive'].replace({'N':1, 'P':2, 'Y':3}, inplace=True)

    X.loc[(X['Mas.Vnr.Type'] == 'None') & (X['Mas.Vnr.Area'] > 1), 'Mas.Vnr.Type'] = 'BrkFace' # most common 
    X.loc[(X['Mas.Vnr.Type'] == 'None') & (X['Mas.Vnr.Area'] == 1), 'Mas.Vnr.Area'] = 0 # 1 sq ft is basically 0
    for vnr_type in X['Mas.Vnr.Type'].unique():
        # so here we set the area equal to the mean of the given veneer type
        X.loc[(X['Mas.Vnr.Type'] == vnr_type) & (X['Mas.Vnr.Area'] == 0), 'Mas.Vnr.Area'] = \
            X[X['Mas.Vnr.Type'] == vnr_type]['Mas.Vnr.Area'].mean() 


    idx_to_drop = X[X['Gr.Liv.Area'] >= 4000].index
    X.drop(idx_to_drop, inplace=True)

    # Total Square Footage
    X['Total.SF'] = X['Total.Bsmt.SF'] + X['Gr.Liv.Area']
    X['Total.Floor.SF'] = X['X1st.Flr.SF'] + X['X2nd.Flr.SF']
    X['Total.Porch.SF'] = X['Open.Porch.SF'] + X['Enclosed.Porch'] + \
        X['X3Ssn.Porch'] + X['Screen.Porch']
        
    # Total Bathrooms
    X['Total.Bathrooms'] = X['Full.Bath'] + .5 * X['Half.Bath'] + \
        X['Bsmt.Full.Bath'] + .5 * X['Bsmt.Half.Bath']

    # Booleans
    X['Has.Basement'] = X['Total.Bsmt.SF'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Garage'] = X['Garage.Area'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Porch'] = X['Total.Porch.SF'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Pool'] = X['Pool.Area'].apply(lambda x: 1 if x > 0 else 0)
    X['Was.Completed'] = (X['Sale.Condition'] != 'Partial').astype(np.int64)

    
    boolean_features = ['Has.Basement', 'Has.Garage', 'Has.Porch', 'Has.Pool', 'Was.Completed']

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    numeric_cols = [f for f in numeric_cols if f not in boolean_features]
    cat_cols = X.select_dtypes(include=['object']).columns

    skew_limit = 0.5
    skew_vals = X[numeric_cols].skew()

    high_skew = skew_vals[abs(skew_vals) > skew_limit]
    skew_cols = high_skew.index.tolist()

    for col in skew_cols:
        X[col] = np.log1p(X[col])
        X[col] = X[col].astype(np.float64)

    X['Overall.Qual'] = X['Overall.Qual'].astype(np.int64)
    X['Overall.Cond'] = X['Overall.Cond'].astype(np.int64)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    numeric_cols = [f for f in numeric_cols if f not in boolean_features]

    cat_cols = X.select_dtypes(include=['object']).columns
    
    X_novo = pd.get_dummies(X, drop_first=True).copy()

    # a partir daqui usar usar os dados origionais para treinar o modelo

    X_model = pd.read_csv('x_model.csv')
    y = pd.read_csv('y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = RobustScaler()
    X_train.loc[:,numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test.loc[:,numeric_cols] = scaler.transform(X_test[numeric_cols])
    X_novo.loc[:,numeric_cols] = scaler.transform(X_novo[numeric_cols])

    params = {'alpha': 6e-05, 'copy_X': True, 'fit_intercept': True, 'max_iter': 50000, 'positive': False, 'precompute': False, 'random_state': None, 'selection': 'cyclic', 'tol': 0.0001, 'warm_start': False}

    lasso = Lasso(**params)
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_novo)
    return y_pred
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import pathlib

RANDOM_SEED = 42

path = pathlib.Path.cwd().parent / 'api'

def remap_categories(
    series: pd.Series,
    old_categories: tuple[str],
    new_category: str,
) -> pd.Series:
    # Add the new category to the list of valid categories.
    series = series.cat.add_categories(new_category)

    # Set all items of the old categories as the new category.
    remapped_items = series.isin(old_categories)
    series.loc[remapped_items] = new_category

    # Clean up the list of categories, the old categories no longer exist.
    series = series.cat.remove_unused_categories()

    return series

def feature_engeneering(df_novo):

    X = df_novo.copy()

    selection = ~(X['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))
    X = X[selection]

    X['MS.Zoning'] = X['MS.Zoning'].cat.remove_unused_categories()

    processed_data = X.copy()

    processed_data['Sale.Type'] = remap_categories(series=processed_data['Sale.Type'],old_categories=('WD ', 'CWD', 'VWD'),new_category='GroupedWD',)
    processed_data['Sale.Type'] = remap_categories(series=processed_data['Sale.Type'],old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),new_category='Other',)

    X = processed_data

    X = X.drop(columns='Street')

    processed_data = X.copy()

    for col in ('Condition.1', 'Condition.2'):
        processed_data[col] = remap_categories(series=processed_data[col],old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),new_category='Railroad',)
        processed_data[col] = remap_categories(series=processed_data[col],old_categories=('Feedr', 'Artery'),new_category='Roads',)
        processed_data[col] = remap_categories(series=processed_data[col],old_categories=('PosA', 'PosN'),new_category='Positive',)
        
    processed_data['Condition'] = pd.Series(index=processed_data.index,dtype=pd.CategoricalDtype(categories=('Norm','Railroad','Roads','Positive','RoadsAndRailroad',)),)
    
    norm_items = processed_data['Condition.1'] == 'Norm'
    processed_data['Condition'][norm_items] = 'Norm'

    railroad_items = (processed_data['Condition.1'] == 'Railroad') & (processed_data['Condition.2'] == 'Norm')
    processed_data['Condition'][railroad_items] = 'Railroad'

    roads_items = (processed_data['Condition.1'] == 'Roads') & (processed_data['Condition.2'] != 'Railroad')
    processed_data['Condition'][roads_items] = 'Roads'

    positive_items = processed_data['Condition.1'] == 'Positive'
    processed_data['Condition'][positive_items] = 'Positive'

    roads_and_railroad_items = ((processed_data['Condition.1'] == 'Railroad') & (processed_data['Condition.2'] == 'Roads')) | ( (processed_data['Condition.1'] == 'Roads') & (processed_data['Condition.2'] == 'Railroad'))
    processed_data['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'
    processed_data = processed_data.drop(columns=['Condition.1', 'Condition.2'])

    X = processed_data

    X['HasShed'] = X['Misc.Feature'] == 'Shed'
    X = X.drop(columns='Misc.Feature')

    X['HasAlley'] = ~X['Alley'].isna()
    X = X.drop(columns='Alley')

    X['Exterior.2nd'] = remap_categories(series=X['Exterior.2nd'],old_categories=('Brk Cmn', ),new_category='BrkComm',)
    X['Exterior.2nd'] = remap_categories(series=X['Exterior.2nd'],old_categories=('CmentBd', ),new_category='CemntBd',)
    X['Exterior.2nd'] = remap_categories(series=X['Exterior.2nd'],old_categories=('Wd Shng', ),new_category='WdShing',)

    for col in ('Exterior.1st', 'Exterior.2nd'):
        categories = X[col].cat.categories
        X[col] = X[col].cat.reorder_categories(sorted(categories))

    processed_data = X.copy()

    mat_count = processed_data['Exterior.1st'].value_counts()
    rare_materials = list(mat_count[mat_count < 40].index)

    processed_data['Exterior'] = remap_categories(series=processed_data['Exterior.1st'],old_categories=rare_materials,new_category='Other',)

    processed_data = processed_data.drop(columns=['Exterior.1st', 'Exterior.2nd'])

    X = processed_data

    X = X.drop(columns='Heating')

    X = X.drop(columns='Roof.Matl')

    X['Roof.Style'] = remap_categories(series=X['Roof.Style'],old_categories=['Flat','Gambrel','Mansard','Shed',],new_category='Other',)

    X['Mas.Vnr.Type'] = remap_categories(series=X['Mas.Vnr.Type'],old_categories=['BrkCmn','CBlock',],new_category='Other',)

    X['Mas.Vnr.Type'] = X['Mas.Vnr.Type'].cat.add_categories('None')

    X['MS.SubClass'] = remap_categories(series=X['MS.SubClass'],old_categories=[75, 45, 180, 40, 150],new_category='Other',)

    X['Foundation'] = remap_categories(series=X['Foundation'],old_categories=['Slab', 'Stone', 'Wood'],new_category='Other',)

    selection = ~X['Neighborhood'].isin(['Blueste','Greens','GrnHill','Landmrk',])
    X = X[selection]

    X['Neighborhood'] = X['Neighborhood'].cat.remove_unused_categories()

    X['Garage.Type'] = X['Garage.Type'].cat.add_categories(['NoGarage'])

    X = X.drop(columns='Utilities')
    X = X.drop(columns='Pool.QC')

    old_categories = list(X['Fence'].cat.categories)
    new_categories = old_categories + ['NoFence']
    X['Fence'] = X['Fence'].cat.set_categories(new_categories)

    X = X.drop(columns='Fireplace.Qu')

    X = X.drop(columns=['Garage.Cond', 'Garage.Qual'])

    X['Garage.Finish'] = X['Garage.Finish'].cat.as_unordered().cat.add_categories(['NoGarage'])

    X['Electrical'][X['Electrical'].isna()] = 'SBrkr'

    X['Bsmt.Exposure'][X['Bsmt.Exposure'].isna()] = 'NA'
    X['Bsmt.Exposure'] = X['Bsmt.Exposure'].cat.as_unordered().cat.remove_unused_categories()

    for col in ('Bsmt.Qual', 'Bsmt.Cond', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
        X[col] = X[col].cat.add_categories(['NA'])
        X[col][X[col].isna()] = 'NA'
        X[col] = X[col].cat.as_unordered().cat.remove_unused_categories()

    X['Bsmt.Cond'][X['Bsmt.Cond'] == 'Po'] = 'Fa'
    X['Bsmt.Cond'][X['Bsmt.Cond'] == 'Ex'] = 'Gd'
    X['Bsmt.Cond'] = X['Bsmt.Cond'].cat.remove_unused_categories()

    garage_age = X['Yr.Sold'] - X['Garage.Yr.Blt'] 
    garage_age[garage_age < 0.0] = 0.0
    X = X.drop(columns='Garage.Yr.Blt')
    X['Garage.Age'] = garage_age

    remod_age = X['Yr.Sold'] - X['Year.Remod.Add']
    remod_age[remod_age < 0.0] = 0.0

    house_age = X['Yr.Sold'] - X['Year.Built']
    house_age[house_age < 0.0] = 0.0

    X = X.drop(columns=['Year.Remod.Add', 'Year.Built'])
    X['Remod.Age'] = remod_age
    X['House.Age'] = house_age

    for col in X.select_dtypes('category').columns:
        X[col] = X[col].cat.remove_unused_categories()

    for col in X.select_dtypes('category'):
        X[col] = X[col].astype(object)

    numeric_cols = X.select_dtypes(include=[np.number]).columns

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

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    X.loc[(X['Mas.Vnr.Type'] == 'None') & (X['Mas.Vnr.Area'] > 1), 'Mas.Vnr.Type'] = 'BrkFace'
    X.loc[(X['Mas.Vnr.Type'] == 'None') & (X['Mas.Vnr.Area'] == 1), 'Mas.Vnr.Area'] = 0 
    for vnr_type in X['Mas.Vnr.Type'].unique():
        X.loc[(X['Mas.Vnr.Type'] == vnr_type) & (X['Mas.Vnr.Area'] == 0), 'Mas.Vnr.Area'] = \
            X[X['Mas.Vnr.Type'] == vnr_type]['Mas.Vnr.Area'].mean() 
    
    X['Total.SF'] = X['Total.Bsmt.SF'] + X['Gr.Liv.Area']
    X['Total.Floor.SF'] = X['X1st.Flr.SF'] + X['X2nd.Flr.SF']
    X['Total.Porch.SF'] = X['Open.Porch.SF'] + X['Enclosed.Porch'] + \
        X['X3Ssn.Porch'] + X['Screen.Porch']
        
    X['Total.Bathrooms'] = X['Full.Bath'] + .5 * X['Half.Bath'] + \
        X['Bsmt.Full.Bath'] + .5 * X['Bsmt.Half.Bath']

    X['Has.Basement'] = X['Total.Bsmt.SF'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Garage'] = X['Garage.Area'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Porch'] = X['Total.Porch.SF'].apply(lambda x: 1 if x > 0 else 0)
    X['Has.Pool'] = X['Pool.Area'].apply(lambda x: 1 if x > 0 else 0)
    X['Was.Completed'] = (X['Sale.Condition'] != 'Partial').astype(np.int64)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    for col in ['Lot.Frontage', 'Lot.Area', 'Lot.Shape', 'Land.Contour', 'Land.Slope', 'Mas.Vnr.Area', 'Exter.Qual', 'Exter.Cond', 'Bsmt.Qual', 'Bsmt.Cond', 'Bsmt.Exposure', 'BsmtFin.SF.1', 'BsmtFin.SF.2', 'Bsmt.Unf.SF', 'Heating.QC', 'X1st.Flr.SF', 'X2nd.Flr.SF', 'Low.Qual.Fin.SF', 'Gr.Liv.Area', 'Bsmt.Full.Bath', 'Bsmt.Half.Bath', 'Half.Bath', 'Kitchen.AbvGr', 'TotRms.AbvGrd', 'Functional', 'Fireplaces', 'Paved.Drive', 'Wood.Deck.SF', 'Open.Porch.SF', 'Enclosed.Porch', 'X3Ssn.Porch', 'Screen.Porch', 'Pool.Area', 'Misc.Val', 'Garage.Age', 'House.Age', 'Total.SF', 'Total.Floor.SF', 'Total.Porch.SF']:
        X[col] = np.log1p(X[col])
        X[col] = X[col].astype(np.float64)

    X['Lot.Shape'] = X['Lot.Shape'].apply(lambda x: float(x))
    X['Land.Contour'] = X['Land.Contour'].apply(lambda x: float(x))
    X['Land.Slope'] = X['Land.Slope'].apply(lambda x: float(x))
    X['Exter.Qual'] = X['Exter.Qual'].apply(lambda x: float(x))
    X['Exter.Cond'] = X['Exter.Cond'].apply(lambda x: float(x))
    X['Bsmt.Qual'] = X['Bsmt.Qual'].apply(lambda x: float(x))
    X['Bsmt.Cond'] = X['Bsmt.Cond'].apply(lambda x: float(x))
    X['Bsmt.Exposure'] = X['Bsmt.Exposure'].apply(lambda x: float(x))
    X['Heating.QC'] = X['Heating.QC'].apply(lambda x: float(x))
    X['Functional'] = X['Functional'].apply(lambda x: float(x))
    X['Paved.Drive'] = X['Paved.Drive'].apply(lambda x: float(x))
    X['Mo.Sold'] = X['Mo.Sold'].apply(lambda x: float(x))
    X['Yr.Sold'] = X['Yr.Sold'].apply(lambda x: float(x))

    if X['Overall.Qual'].isna().any():
        X['Overall.Qual'][0] = 5.0
    if X['Overall.Cond'].isna().any():
        X['Overall.Cond'][0] = 5.0

    X['Overall.Qual'] = X['Overall.Qual'].astype('int64')
    X['Overall.Cond'] = X['Overall.Cond'].astype('int64')

    boolean_features = ['Has.Basement', 'Has.Garage', 'Has.Porch', 'Has.Pool', 'Was.Completed']
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    numeric_cols = [f for f in numeric_cols if f not in boolean_features]

    X_velho = pd.read_csv(path / 'x_model.csv')
    y = pd.read_csv(path / 'y.csv')

    X_velho = pd.get_dummies(X_velho, drop_first=True)
    X = pd.get_dummies(X, drop_first=True)

    #adicionar em X as colunas que não existem em X_velho e ordene as colunas na mesma ordem de X_velho
    for col in X_velho.columns:
        if col not in X.columns:
            X[col] = 0
    
    X = X[X_velho.columns]

    scaler = RobustScaler()
    X_velho.loc[:,numeric_cols] = scaler.fit_transform(X_velho[numeric_cols])
    X.loc[:,numeric_cols] = scaler.transform(X[numeric_cols])

    lasso_params = {'alpha': 8e-05,
                'copy_X': True,
                'fit_intercept': True,
                'max_iter': 50000,
                'positive': False,
                'precompute': False,
                'random_state': None,
                'selection': 'cyclic',
                'tol': 0.0001,
                'warm_start': False}

    lasso = Lasso(**lasso_params)
    lasso.fit(X_velho, y)
    result = lasso.predict(X)


    return 10**result[0]
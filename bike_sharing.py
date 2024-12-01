import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#create the time series data
def create_ts_data(window_size, target_size, df):
    for i in range(window_size -1):
        df[f'users_{i + 1}'] = df['users'].shift(-i - 1)
    for i in range(target_size):
        df[f'target_{i + 1}'] = df['users'].shift(-i - window_size)  
    for i in exogenous:
        for j in range(target_size):
            df[f'{i}_{j + 1}'] = df[i].shift(-j - window_size)
    df = df.drop(exogenous, axis = 1)
    df = df.dropna()
    return df

if __name__ == '__main__':
    df = pd.read_csv('bike-sharing-dataset.csv')    
    df = df.drop('date_time', axis = 1)
    
    exogenous = ['holiday', 'workingday', 'weather', 'temp', 'atemp', 'hum', 'windspeed', 'month', 'hour', 'weekday']
    
    #create the time series data
    window_size = 3
    target_size = 3
    df = create_ts_data(window_size, target_size, df)

    #divide by columns
    x = df.drop([f'target_{i + 1}' for i in range (target_size)], axis = 1)
    y = df[[f'target_{i + 1}' for i in range(target_size)]]
    
    #divide by rows
    length = len(df['target_1'])
    x_train = x.iloc[:int(0.8 * length)]
    x_test = x.iloc[int(0.8 * length):]
    y_train = y.iloc[:int(0.8 * length)]
    y_test = y.iloc[int(0.8 * length):]

    #variables for column scaling
    weather_order = ['rain', 'mist', 'clear']
    num_exogenous = ['temp', 'atemp', 'hum', 'windspeed']
    
    #numbers of models
    models = [LGBMRegressor(force_col_wise = True)] * target_size
    
    for id, model in enumerate(models):
        y_train_cur = y_train[f'target_{id + 1}']
        y_test_cur = y_test[f'target_{id + 1}']
        
        x_train_cur = x_train.copy()
        x_test_cur = x_test.copy()
        
        for i in exogenous:
            for j in range(target_size):
                if(j != id):
                    x_train_cur = x_train_cur.drop(f'{i}_{j + 1}', axis = 1)
                    x_test_cur = x_test_cur.drop(f'{i}_{j + 1}', axis = 1)
        
        #preprocess the data
        numeric_features = ['users'] + [f'users_{i}' for i in range(1, window_size)] + [f'{i}_{id + 1}' for i in num_exogenous]

        preprocess = ColumnTransformer(transformers=[('numerical', StandardScaler(), numeric_features),
                                                     ('ordinal', OrdinalEncoder(categories = [weather_order]), [f'weather_{id + 1}'])], 
                                       remainder='passthrough')
        x_train_cur = preprocess.fit_transform(x_train_cur)
        x_test_cur = preprocess.transform(x_test_cur)
        
        #train the model
        model = model.fit(x_train_cur, y_train_cur)
        y_pred = model.predict(x_test_cur)
        print(f'R2: {r2_score(y_test_cur, y_pred)}')
        print(f'MSE: {mean_squared_error(y_test_cur, y_pred)}')
        print(f'MAE: {mean_absolute_error(y_test_cur, y_pred)}')
        print('-------------------------------------------------')



    



    
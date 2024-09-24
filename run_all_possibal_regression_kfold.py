import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
from sklearn.model_selection import KFold
import sys

random_state = 42
n_k_fold = 5

REPLACE_LOG_15PRAMS = True
if int(sys.argv[1]) == 15:
    REPLACE_LOG_15PRAMS = True
elif int(sys.argv[1]) == 24:
    REPLACE_LOG_15PRAMS = False

def group_by_length(lst):
    result = {}
    for length in range(1, len(lst) + 1):
        combinations = list(itertools.combinations(lst, length))
        result[length] = combinations
    return result

def evaluate(y_test,y_pred_test,y_train,y_pred_train,X_test):
    # 5. Evaluate the Model
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Calculate RMSE
    rmse_test = np.sqrt(mse_test)
    rmse_train = np.sqrt(mse_train)

    # Calculate BIC and AIC
    n = len(y_test)  # Number of observations
    k = X_test.shape[1]   # Number of predictors (features)
    bic_test = n * np.log(mse_test) + k * np.log(n)
    aic_test = n * np.log(mse_test) + 2 * k

    n = len(y_train)  # Number of observations
    k = X_test.shape[1]   # Number of predictors (features)
    bic_train = n * np.log(mse_train) + k * np.log(n)
    aic_train = n * np.log(mse_train) + 2 * k
    
    result = {
        'mse_test' : mse_test,
        'rmse_test' : rmse_test,
        'mae_test' : mae_test,
        'r2_test' : r2_test,
        'bic_test' : bic_test,
        'aic_test' : aic_test,
        'mse_train' : mse_train,
        'rmse_train' : rmse_train,
        'mae_train' : mae_train,
        'r2_train' : r2_train,
        'bic_train' : bic_train,
        'aic_train' : aic_train,    
    }

    return result

def train_regression_with_columns(selected_columns,X,y,random_state):
    # 1. Select columns
    X = X[selected_columns]

    # 2. Split the Data
    kf = KFold(n_splits=n_k_fold, shuffle=True, random_state=random_state)

    results = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[list(train_index)], X.loc[list(test_index)]
        y_train, y_test = y.loc[list(train_index)], y.loc[list(test_index)]

        # 3. Create and Train the Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 4. Make Predictions
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        result = evaluate(y_test,y_pred_test,y_train,y_pred_train,X_test)
        results.append(result)

    # Combine values for each key into lists
    combined_result = {key: [d[key] for d in results] for key in results[0]}
    combined_result_avg = {f'{key}_avg': sum(values) / len(values) for key, values in combined_result.items()}
    
    return {**combined_result, **combined_result_avg}

#### Prepare Data
df = pd.read_csv("data/raw_data_phawit_revised_init.csv")
selected_columns_15_and_y = ['raw_g_dflow','raw_g_mflow','raw_g_depth','raw_g_pntu','res_weight','distrib_area','distrib_main','distrib_storage','distrib_press','distrib_lost','calc_flow','calc_NTU_avg','calc_NTU_peak','calc_elev_change','calc_hp','total_energy_use_kBtu']
df = df[selected_columns_15_and_y]

# List of columns to take the logarithm of
columns_to_log = ['raw_g_dflow', 'raw_g_mflow','res_weight','distrib_area','distrib_main','distrib_storage','calc_flow','calc_NTU_peak','calc_hp','total_energy_use_kBtu']  
for column in columns_to_log:
    new_column_name = f'ln_{column}'
    df[new_column_name] = df[column].apply(lambda x: np.log1p(x))

if REPLACE_LOG_15PRAMS:
    df = df.drop(columns=columns_to_log)
    X = df.drop(columns=['ln_total_energy_use_kBtu'])  # All columns except the target
    y = df['ln_total_energy_use_kBtu']
else:
    X = df.drop(columns=['total_energy_use_kBtu','ln_total_energy_use_kBtu'])  # All columns except the target
    y = df['ln_total_energy_use_kBtu']

print('='*100)
print('X.shape:: ',X.shape)
print('y.shape::',y.shape)
print('='*100)

### Loop training

selected_columns = list(X.columns)
grouped = group_by_length(selected_columns)
i = 0
t = time.time()
for length, groups in grouped.items():
    print(f"\nLength {length}",'='*100)
    G = [list(x) for x in groups]

    for ii,g in enumerate(G):
        
        print(f"\nround={i} n_columns={length}::[{ii+1}/{len(G)}]::{g}")
        results = train_regression_with_columns(g,X,y,random_state)
        results['n_columns'] = length
        results['random_state'] = random_state
        results['features'] = g
        results['round'] = i
        results['no'] = ii
        results['time'] = time.time() - t

        t = time.time()

        print('results:',results)
        print('\n','-'*100)
        
        # Save results to CSV
        results_df = pd.DataFrame([results])
        if REPLACE_LOG_15PRAMS:
            ft = 15
        else:
            ft = 24
        file_path = f'results_all_possible_regression/feature{ft}/randomstate{random_state}/features_{length}.csv'
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            results_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(file_path, mode='w', header=True, index=False)
    
        i += 1
        
        
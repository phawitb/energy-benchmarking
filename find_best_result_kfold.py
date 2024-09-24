import os
import pandas as pd

result_floder = 'results_all_possible_regression'

for all_feature in os.listdir(result_floder):
    for random_state in os.listdir(f"{result_floder}/{all_feature}"):
        for n_feature in os.listdir(f"{result_floder}/{all_feature}/{random_state}"):
            # print('random_state',random_state)
            # print('all_feature:',all_feature)
            # print('n_feature',n_feature.split('_')[-1].split('.')[0])

            print(f'results_all_possible_regression/{all_feature}/{random_state}/{n_feature}')

            df = pd.read_csv(f'results_all_possible_regression/{all_feature}/{random_state}/{n_feature}')
            filtered_df = df[df['mse_test_avg'] == df['mse_test_avg'].min()]
            df_dict = filtered_df.to_dict(orient='index')
            min_data = df_dict[list(df_dict.keys())[0]]
            min_data['all_feature'] = all_feature
            min_data['n_feature'] = int(n_feature.split('_')[-1].split('.')[0])
            
            print(min_data)

            # Create directory if it doesn't exist
            directory = f'results_all_possible_regression/best'
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save results to CSV
            results_df = pd.DataFrame([min_data])
            file_path = f'{directory}/best_{all_feature}_{random_state}.csv'
            if os.path.exists(file_path):
                results_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(file_path, mode='w', header=True, index=False)

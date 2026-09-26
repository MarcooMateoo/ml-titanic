import pandas as pd 
import os
    
def save_load_model_hist(algo, cv_scores, description, clear_model_history = False):
    
    file_name = '../outputs/logs/experiment_log.csv'
    exp_log_cols = ['Model', 'Algo', 'Description', "Mean", "Std", "Kaggle Score"]
    last_algo_stat = pd.DataFrame(columns=exp_log_cols)

    if(os.path.exists(file_name)):
        exp_log = pd.read_csv(file_name)
    else: 
        exp_log = pd.DataFrame(columns = exp_log_cols)
        
    if clear_model_history:
        exp_log = pd.DataFrame(columns = exp_log_cols) 
        last_model_no = 0
    else: 
        if len(exp_log) == 0:
            last_model_no = 0
        else:
            last_model_no = (exp_log['Model'].str 
                            .split('V_') 
                            .str[1] 
                            .astype(int) 
                            .max())
            last_algo_stat = exp_log[exp_log['Algo'] == algo].sort_values(by='Mean', ascending=False).head(1)


    model_score = {
            'Model': ["V_" + str(last_model_no + 1)],
            'Algo': algo,
            'Description': [description],
            'Mean': [cv_scores.mean()],
            'Std': [cv_scores.std()],
            'Kaggle Score' : 0
    }

    new_model = pd.DataFrame(model_score)

    if len(exp_log) == 0:
        exp_log = new_model
    else:
        exp_log = pd.concat([exp_log, new_model], axis = 0)
    exp_log.to_csv(file_name, index=False)

    return new_model, last_algo_stat,  exp_log
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import time
from func import *
np.random.seed(42)

#df = pd.read_csv('scorer_dataset_without_disqualified_candidates.zip', compression='zip')
#df = df[['you_oid','competitor_id','you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host','color_score','openai_score','image_score','mobilenet_score','Exact Match']]

#train, test = get_train_test(df)
#train.to_csv('train_wo_dbt.csv', index=False)
#test.to_csv('test_wo_dbt.csv', index=False)

train = pd.read_csv('train_wo_dbt.csv')
train = train[['you_oid','competitor_id','you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host','color_score','openai_score','image_score','mobilenet_score','Exact Match']]

X_train, y_train = preprocessing(train)
X_train_encoded, _ = cat_encoding(X_train, y_train, fit=True, encoder=None)

#X_test, y_test = preprocessing(test)
#X_test_encoded, _ = cat_encoding(X_test, fit=False, encoder=target_encoder)

print(X_train_encoded.shape, y_train.shape) #,X_test_encoded.shape, X_test.shape, y_test.shape)

def objective(trial):
    params = {
        'n_estimators': 200,
        'max_depth': trial.suggest_int('max_depth', 4, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3),
        'subsample': trial.suggest_float('subsample', 0.8, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.7),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 0.7),
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10),
        'eta': trial.suggest_float('eta', 0.8, 1),
        'objective': 'binary:logistic',
        'random_state': 42,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    }

    params_old = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'subsample': 0.56,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 4, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'eta': trial.suggest_float('eta', 0.1, 1),
        'objective': 'binary:logistic',
        'seed': 42
    }

    model = xgb.XGBClassifier(**params)  
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_encoded, y_train, cv=skf, scoring='f1')
    f1_score = scores.mean()
    return f1_score

# Assuming you have your training data (X_train, y_train) prepared
start_time = time.time()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300, show_progress_bar=True)
print("Time taken:", time.time() - start_time)

best_params = study.best_params
best_value = study.best_value

print("Best parameters:", best_params)
print("Best value:", best_value)

# save best hyperparameters to a text file
with open('best_hyperparams_xgb.txt', 'w') as f:
    f.write(str(best_params))

# save the study object to a pickle file
with open('study_xgb.pkl', 'wb') as f:
    pickle.dump(study, f)
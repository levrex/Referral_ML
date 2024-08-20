from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
import optuna
import pickle
from sklearn.metrics import average_precision_score
from combat.pycombat import pycombat
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import joblib


#xgb.config_context(verbosity= 1) # silent
optuna.logging.set_verbosity(optuna.logging.INFO) # optuna.logging.INFO
 
TARGET = 'RA' #'OA', 'Chronic', 'RA', 'FMS', Arthralgia

# Sample data
# Import data

df_train = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/extra/df_train.csv')
df_test = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/extra/df_test.csv')

#df_train = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/extra/df_train_RA.csv')
#df_test = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/extra/df_test_RA.csv')

X_train = df_train['fixedLine']
X_test = df_test['fixedLine']

y_train = df_train[TARGET]#.replace({0: False, 1: True})
y_test = df_test[TARGET]#.replace({0: False, 1: True})


def objective_old(trial, target='Chronic'):
    
    # Import data
    df_train = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/extra/df_train.csv')
    
    y = df_train[target].replace({0: False, 1: True})
    # Bookmark all predictions
    y_pred = []
    
    # Perform kfold CV
    # Apply 5 fold CV
    kf = KFold(n_splits=5) 
    
    for train_index, test_index in kf.split(df_train['fixedLine']):
        
        #print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
        train_x, test_x = df_train['fixedLine'].loc[train_index].values, df_train['fixedLine'].loc[test_index].values
        train_y, test_y = np.take(y, np.array(train_index)), np.take(y, np.array(test_index))

        tfidf = TfidfVectorizer(max_df=0.9, min_df=0.01, ngram_range=(1,3)) 
        X_feat = tfidf.fit_transform(train_x)
        X_feat_test = tfidf.transform(test_x)

        #pipeline.fit_transform(train_x, test_x)

        dtrain = xgb.DMatrix(X_feat, label=train_y)
        dtest = xgb.DMatrix(X_feat_test, label=test_y)

        param = {
            'objective': "binary:logistic",
            'num_class':1,
            "eval_metric": "logloss", # aucpr
            "booster": trial.suggest_categorical("booster", ["gbtree"]), # only gbtree , "dart"
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            #"n_trees": 50,
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], verbose_eval=1) 
        preds = bst.predict(dtest)
        
        # Create a confusion matrix over all data!
        y_pred.extend(bst.predict(dtest))

    auc_pr = roc_auc_score(y, y_pred)
    print(trial._trial_id)
    return auc_pr # precision recall curve

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective_old(trial, TARGET), n_trials=1000) # test # 2000 # -1 -> bootstrapping (take random 100 samples) , n_jobs=1
print('Best trial: %s' % study.best_trial.number)
print('Performance (auc): %s' % study.best_trial.value)
print('Corresponding Parameters: %s' % study.best_trial.params)


import plotly as py
# optuna.visualization.plot_intermediate_values(study)
fig = optuna.visualization.plot_optimization_history(study)
py.offline.plot(fig, filename='/exports/reum/tdmaarseveen/gitlab/referral_ml/figures/tuning/hyperparamtuning_optimization_%s_ngram_1000iter_test.html' % (TARGET), auto_open=False)


# Train the final model with the best hyperparameters
best_params = study.best_trial.params

best_params['objective'] = "binary:logistic"
best_params['num_class'] = 1
best_params['eval_metric'] = "logloss"
print(best_params)

tfidf = TfidfVectorizer(max_df=0.9, min_df=0.01, ngram_range=(1,3)) # 0.01
X_feat = tfidf.fit_transform(X_train)
X_feat_test = tfidf.transform(X_test)

dtrain = xgb.DMatrix(X_feat, label=y_train)
dtest = xgb.DMatrix(X_feat_test, label=y_test)

bst = xgb.train(params = best_params, dtrain=dtrain, evals=[(dtest, "validation")], verbose_eval=1)
preds = bst.predict(dtest)

with open('/exports/reum/tdmaarseveen/gitlab/referral_ml/model/tfidf/tfidf_vectorizer_%s_ngram_1000iter_test.pk' % TARGET, 'wb') as fin:
    pickle.dump(tfidf, fin)
    
with open('/exports/reum/tdmaarseveen/gitlab/referral_ml/model/xgb/xgb_%s_ngram_1000iter_test.pk' % TARGET, 'wb') as fin:
    pickle.dump(bst, fin)

auc_roc = roc_auc_score(y_test, preds)
print('Test set auc:', auc_roc)


from src.data_processing.data_processing import process_values
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
import lightgbm as lgb
import optuna
import pandas as pd
import joblib

TRAIN_VALUES_FILEPATH = "data/train_values.csv"
TRAIN_LABELS_FILEPATH = "data/train_labels.csv"

raw_df = pd.read_csv(TRAIN_VALUES_FILEPATH)
raw_df.index = raw_df['building_id']
raw_df.drop('building_id', axis=1, inplace=True)
labels = pd.read_csv(TRAIN_LABELS_FILEPATH)
labels.index = labels['building_id']
labels.drop('building_id', axis=1, inplace=True)

df = process_values(raw_df)

X_train, X_test, y_train, y_test = train_test_split(df,
                                                    labels.damage_grade,
                                                    test_size=.3,
                                                    random_state=69,
                                                    stratify=labels.damage_grade)

# Setup a lightgbm with optuna
#categorical_features = [c for c, col in enumerate(df.columns) if 'cat' in col]
#train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False, reference=train_data)


def objective(trial):
    op_parameters = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 5, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.9),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 8),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.2),
        'verbose': 0
    }

    op_model = lgb.train(op_parameters,
                         train_data,
                         valid_sets=test_data,
                         num_boost_round=100,
                         early_stopping_rounds=5)

    op_loss = op_model.best_score['valid_0']['rmse']

    return op_loss

study = optuna.create_study()
study.optimize(objective, n_trials=50)

parameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'num_leaves': study.best_params['num_leaves'],
    'bagging_fraction': study.best_params['bagging_fraction'],
    'feature_fraction': study.best_params['feature_fraction'],
    'bagging_freq': study.best_params['bagging_freq'],
    'learning_rate': study.best_params['learning_rate'],
    'verbose': 0
}

lgb_model = lgb.train(parameters,
                      train_data,
                      valid_sets=test_data,
                      num_boost_round=20,
                      early_stopping_rounds=5)

lgb_preds = lgb_model.predict(X_test)

print(r2_score(y_test, lgb_preds))

rounded_preds = [round(x) for x in lgb_preds]
print(f1_score(y_true=y_test, y_pred=rounded_preds, average='micro'))


correct = 0
incorrect = 0
for i in range(len(y_test)):
    if rounded_preds[i] == y_test.values[i]:
        correct += 1
    else:
        incorrect += 1

print("Correct: {0}, incorrect: {1}".format(correct, incorrect))
print("Accuracy: {0}".format(correct/(correct + incorrect)))

# Confusion matrix
data = {"y_pred": rounded_preds, "y_test": y_test.to_list()}
conf_df = pd.DataFrame(data, columns=["y_pred", "y_test"])
confusion_matrix = pd.crosstab(conf_df['y_test'], conf_df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Store model
joblib.dump(lgb_model, "models/lgb_210409b.pkl")

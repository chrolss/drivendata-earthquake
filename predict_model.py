import joblib
import pandas as pd
from src.data_processing.data_processing import process_values

model = joblib.load("models/lgb_210409b.pkl")

data = pd.read_csv("data/test_values.csv")

data_processed = process_values(data)

building_ids = data_processed['building_id']

preds = model.predict(data_processed.drop('building_id', axis=1))

rounded_preds = [round(x) for x in preds]

submission_dict = {
    'building_id': building_ids,
    'damage_grade': rounded_preds
}
submission = pd.DataFrame(data=submission_dict, columns=['building_id', 'damage_grade'])

submission.to_csv("data/submissions/submission002.csv", index=False)

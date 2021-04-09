import pandas as pd
import joblib


def _geo_level_assignment(x, geo):
    if x in geo.keys():
        return geo[x]
    else:
        return 0


def _geo_risk_assignment(lvl_id, grade_nr, geo):
    try:
        return geo.loc[lvl_id, grade_nr]
    except KeyError:
        return 0.5


def process_values(values):
    # Load the geo level dictionaries
    geo_level_1 = joblib.load("data/geo_level_1.pkl")
    geo_level_2 = joblib.load("data/geo_level_2.pkl")
    geo_level_3 = joblib.load("data/geo_level_3.pkl")

    # Load the risk level dictionaries
    geo_level_1_risks = joblib.load("data/geo_level_1_risks.pkl")
    geo_level_2_risks = joblib.load("data/geo_level_2_risks.pkl")
    geo_level_3_risks = joblib.load("data/geo_level_3_risks.pkl")
    geo_level_1_risks = geo_level_1_risks.fillna(0)
    geo_level_2_risks = geo_level_2_risks.fillna(0)
    geo_level_3_risks = geo_level_3_risks.fillna(0)

    # Pre-defined categorical features -> Set to category type for pd.get_dummies()
    cat_cols = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
                'position', 'plan_configuration', 'legal_ownership_status']
    for col in cat_cols:
        values[col] = values[col].astype("category")

    df = pd.get_dummies(values)

    # Add the engineered feature "geo_level_x_grade_y"
    df['geo_lvl_1_grade_1'] = df['geo_level_1_id'].apply(lambda x: _geo_level_assignment(x, geo_level_1[0]))
    df['geo_lvl_1_grade_2'] = df['geo_level_1_id'].apply(lambda x: _geo_level_assignment(x, geo_level_1[1]))
    df['geo_lvl_1_grade_3'] = df['geo_level_1_id'].apply(lambda x: _geo_level_assignment(x, geo_level_1[2]))
    df['geo_lvl_2_grade_1'] = df['geo_level_2_id'].apply(lambda x: _geo_level_assignment(x, geo_level_2[0]))
    df['geo_lvl_2_grade_2'] = df['geo_level_2_id'].apply(lambda x: _geo_level_assignment(x, geo_level_2[1]))
    df['geo_lvl_2_grade_3'] = df['geo_level_2_id'].apply(lambda x: _geo_level_assignment(x, geo_level_2[2]))
    df['geo_lvl_3_grade_1'] = df['geo_level_3_id'].apply(lambda x: _geo_level_assignment(x, geo_level_3[0]))
    df['geo_lvl_3_grade_2'] = df['geo_level_3_id'].apply(lambda x: _geo_level_assignment(x, geo_level_3[1]))
    df['geo_lvl_3_grade_3'] = df['geo_level_3_id'].apply(lambda x: _geo_level_assignment(x, geo_level_3[2]))

    # Add the risk columns
    df['geo_level_1_risk_1'] = df['geo_level_1_id'].apply(lambda x: _geo_risk_assignment(x, 1, geo_level_1_risks))
    df['geo_level_1_risk_2'] = df['geo_level_1_id'].apply(lambda x: _geo_risk_assignment(x, 2, geo_level_1_risks))
    df['geo_level_1_risk_3'] = df['geo_level_1_id'].apply(lambda x: _geo_risk_assignment(x, 3, geo_level_1_risks))
    df['geo_level_2_risk_1'] = df['geo_level_2_id'].apply(lambda x: _geo_risk_assignment(x, 1, geo_level_2_risks))
    df['geo_level_2_risk_2'] = df['geo_level_2_id'].apply(lambda x: _geo_risk_assignment(x, 2, geo_level_2_risks))
    df['geo_level_2_risk_3'] = df['geo_level_2_id'].apply(lambda x: _geo_risk_assignment(x, 3, geo_level_2_risks))
    df['geo_level_3_risk_1'] = df['geo_level_3_id'].apply(lambda x: _geo_risk_assignment(x, 1, geo_level_3_risks))
    df['geo_level_3_risk_2'] = df['geo_level_3_id'].apply(lambda x: _geo_risk_assignment(x, 2, geo_level_3_risks))
    df['geo_level_3_risk_3'] = df['geo_level_3_id'].apply(lambda x: _geo_risk_assignment(x, 3, geo_level_3_risks))

    df.drop('geo_level_1_id', axis=1, inplace=True)
    df.drop('geo_level_2_id', axis=1, inplace=True)
    df.drop('geo_level_3_id', axis=1, inplace=True)

    return df


data = pd.read_csv("data/train_values.csv")
df = process_values(data)

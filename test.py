import pandas as pd 
import numpy as np 
import os 
import random
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")

sample_df = pd.read_csv(os.path.join(data_dir, "MSampleSubmissionStage1.csv"))
sample_df["Season"] = sample_df["ID"].apply(lambda x: int(x.split('_')[0]))
sample_df["TeamIdA"] = sample_df["ID"].apply(lambda x: int(x.split('_')[1]))
sample_df["TeamIdB"] = sample_df["ID"].apply(lambda x: int(x.split('_')[2]))

# Load data
df = pd.read_csv("training_data.csv")
features = ["rank", "seed", "adj_em", "adj_o", "adj_d", "adj_t"
                , "luck", "sos_em", "sos_o", "sos_d", "ncsos_em", "wins"
                , "losses", "win_pct", "home_win_pct", "away_win_pct", "power_six"
                , "n_champs", "n_ffour", "n_eeight", "home", "home_proxy", "massey_rank"]
param = {'num_leaves': 64
        , 'objective': 'binary'
        , 'metric': ['auc', 'binary_logloss']
        , 'seed': 42}
team_vectors = pd.read_csv("team_vectors.csv")
print(team_vectors.columns)
# split dataset
def split_data(df, fraction=0.1):
    valid_rows = int(len(df) * fraction)
    train = df[:-valid_rows * 2]
    valid = df[-valid_rows * 2:-valid_rows]
    test = df[-valid_rows:]

    return train, valid, test

# scale features
def scale_features(train, valid, test, features):
    min_ = train[features].min()
    max_ = train[features].max()

    train[features] = (train[features] - min_) / (max_ - min_)
    valid[features] = (valid[features] - min_) / (max_ - min_)

    if test is not None:
        test[features] = (test[features] - min_) / (max_ - min_)
        return train, valid, test
    else:
        return train, valid

# train model
def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(["season", "WTeamID", "LTeamID", "result"])
    dtrain = lgb.Dataset(train[feature_cols], label=train["result"])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid["result"])

    num_round = 100
    print("Training model!")
    bst = lgb.train(param
                    , dtrain
                    , num_round
                    , valid_sets=[dvalid]
                    , early_stopping_rounds=20
                    , verbose_eval=False)
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid["result"], valid_pred)
    valid_logloss = metrics.log_loss(valid["result"], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    print(f"Validation log loss: {valid_logloss}")

    print(dict(zip(feature_cols, bst.feature_importance())))

    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test["result"], test_pred)
        test_logloss = metrics.log_loss(test["result"], test_pred)
        print(f"Test AUC score: {test_score}")
        print(f"Test log loss: {test_logloss}")
        return bst, valid_score, test_score
    else:
        return bst, valid_score

def train_classifier(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(["season", "WTeamID", "LTeamID", "result"])
    model = lgb.LGBMClassifier(**param, n_estimators=1000)
    train_x, train_y = train[feature_cols], train["result"]
    valid_x, valid_y = valid[feature_cols], valid["result"]
    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='logloss', early_stopping_rounds=100)
    valid_pred = model.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid["result"], valid_pred)
    valid_loss = metrics.log_loss(valid["result"], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    print(f"Validation log loss: {valid_loss}")
    if test is not None:
        test_pred = model.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test["result"], test_pred)
        test_loss = metrics.log_loss(test["result"], test_pred)
        print(f"Test AUC score: {test_score}")
        print(f"Test log loss: {test_loss}")
    
    return model, feature_cols


# sample test
def sample_test(TeamIdA, TeamIdB, model, year=2021, feature_cols=None, from_indy=False):
    low_id = min(TeamIdA, TeamIdB)
    high_id = max(TeamIdA, TeamIdB)
    if feature_cols is None:
        feature_cols = team_vectors.columns  
    if from_indy:
        feature_cols = ["home_proxy" if c=="home" else c for c in feature_cols]
    vector_a = team_vectors[(team_vectors["teamID"] == low_id) & (team_vectors["season"] == year)][feature_cols].to_numpy()
    vector_b = team_vectors[(team_vectors["teamID"] == high_id) & (team_vectors["season"] == year)][feature_cols].to_numpy()
    diff = [a - b for a, b in zip(vector_a[0], vector_b[0])]
    diff = np.array(diff).reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        pred = model.predict_proba(diff)[0]
    pred = model.predict(diff)[0]
    home = diff[0][-1]
    #print(f"In the {year} season, Team {low_id} has a {pred[0][1]*100}% chance of winning")
    return pred, home

def generateSample(feature_cols):
    result_ls = []
    for i, row in sample_df.iterrows():
        prob, home = sample_test(row["TeamIdA"], row["TeamIdB"], bst, row["Season"], feature_cols)
        pred, _ = sample_test(row["TeamIdA"], row["TeamIdB"], model, row["Season"], feature_cols)
        if prob < 0.67 and prob > 0.5 and pred == 1:
            prob = 0.67
        elif prob < 0.67 and prob > 0.5 and pred == 0:
            if home >= 0:
                prob = 0.67
            else:
                prob = 0.33
        elif prob > 0.33 and prob <= 0.5 and pred == 0:
            prob = 0.33
        elif prob > 0.33 and prob <= 0.5 and pred == 1:
            if home >= 0:
                prob = 0.67
            else:
                prob = 0.33
        elif prob <= 0.33 and pred == 1:
            if home >= 0:
                prob = 0.67
            else:
                prob = 0.33
        elif prob >= 0.67 and pred == 0:
            if home >= 0:
                prob = 0.67
            else:
                prob = 0.33
        elif prob >= 0.95:
            prob = 0.95
        elif prob <= 0.05:
            prob = 0.05
        else:
            prob = prob

        d = {"ID": row["ID"], 
            "Pred": prob
        }
        result_ls.append(d)

    results = pd.DataFrame(result_ls)
    print(results)
    results.to_csv("results_step1.csv", index=False)

feats = ["adj_em", "luck", "sos_em", "ncsos_em", "home", "massey_rank"]
train, valid, test = split_data(df)
#train, valid, test = scale_features(train, valid, test, features)
bst, _, _ = train_model(train, valid, test, feats)
model, features = train_classifier(train, valid, test, feats)
#sample_test(1438, 1437, model, 2021, feature_cols=feats, from_indy=False)
generateSample(feats)
import pandas as pd 
import numpy as np 
import os 
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")
"""
test_df = pd.read_csv(os.path.join(data_dir, "MSampleSubmissionStage1.csv"))
test_df["Season"] = test_df["ID"].apply(lambda x: int(x.split('_')[0]))
test_df["TeamIdA"] = test_df["ID"].apply(lambda x: int(x.split('_')[1]))
test_df["TeamIdB"] = test_df["ID"].apply(lambda x: int(x.split('_')[2]))
print(test_df)
"""
# Load data
df = pd.read_csv("training_data.csv")
features = ["rank", "seed", "adj_em", "adj_o", "adj_d", "adj_t"
                , "luck", "sos_em", "sos_o", "sos_d", "ncsos_em", "wins"
                , "losses", "win_pct", "home_win_pct", "away_win_pct", "power_six"
                , "n_champs", "n_ffour", "n_eeight", "home"]
team_vectors = pd.read_csv("team_vectors.csv")
print(df)
print(team_vectors)

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

    param = {'num_leaves': 64
            , 'objective': 'binary'
            , 'metric': ['auc', 'logloss']
            , 'seed': 42}
    num_round = 1000
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

    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test["result"], test_pred)
        test_logloss = metrics.log_loss(test["result"], test_pred)
        print(f"Test AUC score: {test_score}")
        print(f"Test log loss: {test_logloss}")
        return bst, valid_score, test_score
    else:
        return bst, valid_score

"""
train, valid, test = split_data(df)
train, valid, test = scale_features(train, valid, test, features)
_ = train_model(train, valid)
"""
# sample test
def sample_test(TeamIdA, TeamIdB):
    print(team_vectors)
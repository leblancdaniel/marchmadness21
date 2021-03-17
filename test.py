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

sample_df = pd.read_csv(os.path.join(data_dir, "MSampleSubmissionStage2.csv"))
sample_df["Season"] = sample_df["ID"].apply(lambda x: int(x.split('_')[0]))
sample_df["TeamIdA"] = sample_df["ID"].apply(lambda x: int(x.split('_')[1]))
sample_df["TeamIdB"] = sample_df["ID"].apply(lambda x: int(x.split('_')[2]))

# Load data
df = pd.read_csv("training_data.csv")

param = {'num_leaves': 64
        , 'objective': 'binary'
        , 'metric': ['auc', 'binary_logloss']
        , 'seed': 4}
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
    
    return model

# sample test
def sample_test(TeamIdA, TeamIdB, model, year=2021, feature_cols=None, from_indy=False):
    low_id = min(TeamIdA, TeamIdB)
    high_id = max(TeamIdA, TeamIdB)
    if feature_cols is None:
        feature_cols = team_vectors.columns.drop(["season", "teamID"])
    if from_indy:
        feature_cols = ["home_proxy" if c=="home" else c for c in feature_cols]
    vector_a = team_vectors[(team_vectors["teamID"] == low_id) & (team_vectors["season"] == year)][feature_cols].to_numpy()
    vector_b = team_vectors[(team_vectors["teamID"] == high_id) & (team_vectors["season"] == year)][feature_cols].to_numpy()
    diff = [a - b for a, b in zip(vector_a[0], vector_b[0])]
    diff = np.array(diff).reshape(1, -1)

    if hasattr(model, 'predict_proba'):
        pred = model.predict_proba(diff)[0]
    pred = model.predict(diff)[0]
    luck = diff[0][1]
    #print(f"In the {year} season, Team {low_id} has a {pred[0][1]*100}% chance of winning")
    teams = [low_id, high_id]
    return pred, luck, teams

def generateSample(feature_cols, from_indy=False):
    result_ls = []
    for i, row in sample_df.iterrows():
        prob, luck, _ = sample_test(row["TeamIdA"], row["TeamIdB"], bst, row["Season"], feature_cols, from_indy)
        pred, _, _ = sample_test(row["TeamIdA"], row["TeamIdB"], model, row["Season"], feature_cols, from_indy)

        if prob < 0.64 and prob > 0.36:
            r = random.randint(0, 1)
            if r == 0:
                prob = 0.36
            else:
                prob = 0.64
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
    results.to_csv("random_results_step2.csv", index=False)
    return results

def sim_winner(teamA, teamB, df):
    low = min(teamA, teamB)
    high = max(teamA, teamB)
    game_id = "2021_" + str(low) + "_" + str(high)
    pred = df.loc[df["ID"] == game_id, "Pred"].iloc[0]
    if pred < 0.5:
        print(high, "is the expected winner")
        return high
    else:
        print(low, "is the expected winner")
        return low

def simulation(df):
    print("-----Simulate Pre-Round Seeds-----")
    west_sixteen = sim_winner(1111, 1313, df)          # NORF vs App St
    west_eleven = sim_winner(1179, 1455, df)           # Drake vs Wichita St
    east_sixteen = sim_winner(1291, 1411, df)          # Mt St Mary vs TX Southern
    east_eleven = sim_winner(1277, 1417, df)           # Michigan St vs UCLA
    print("-----Simulate First Round-----")
    print("WEST")
    west_32_a = sim_winner(1211, west_sixteen, df)     # Gonzaga vs NORF/APP
    west_32_b = sim_winner(1328, 1281, df)             # Oklahoma vs Missouri
    west_32_c = sim_winner(1166, 1364, df)             # Creighton vs UCSB
    west_32_d = sim_winner(1438, 1325, df)             # Virginia vs Ohio
    west_32_e = sim_winner(1425, west_eleven, df)      # USC vs WICH/DRKE
    west_32_f = sim_winner(1242, 1186, df)             # Kansas vs E Washington
    west_32_g = sim_winner(1332, 1433, df)             # Oregon vs VCU
    west_32_h = sim_winner(1234, 1213, df)             # Iowa vs Grand Canyon
    print("SOUTH")
    south_32_a = sim_winner(1124, 1216, df)            # Baylor vs Hartford
    south_32_b = sim_winner(1314, 1458, df)            # N Carolina vs Wisconsin
    south_32_c = sim_winner(1437, 1457, df)            # Villanova vs Winthrop
    south_32_d = sim_winner(1345, 1317, df)            # Purdue vs N Texas
    south_32_e = sim_winner(1403, 1429, df)            # Texas Tech vs Utah St
    south_32_f = sim_winner(1116, 1159, df)            # Arkansas vs Colgate
    south_32_g = sim_winner(1196, 1439, df)            # Florida vs Virginia Tech
    south_32_h = sim_winner(1326, 1331, df)            # Ohio St vs Oral Roberts
    print("EAST")
    east_32_a = sim_winner(1276, east_sixteen, df)     # Michigan vs MSM/TXSO
    east_32_b = sim_winner(1261, 1382, df)             # LSU vs St Bonaventure
    east_32_c = sim_winner(1160, 1207, df)             # Colorado vs Georgetown
    east_32_d = sim_winner(1199, 1422, df)             # Florida St vs UNC Greensboro
    east_32_e = sim_winner(1140, east_eleven, df)      # BYU vs MSU/UCLA
    east_32_f = sim_winner(1400, 1101, df)             # Texas vs Abil Christian
    east_32_g = sim_winner(1163, 1268, df)             # UConn vs Maryland
    east_32_h = sim_winner(1104, 1233, df)             # Alabama vs Iona
    print("MIDWEST")
    midwest_32_a = sim_winner(1228, 1180, df)          # Illinois vs Drexel
    midwest_32_b = sim_winner(1260, 1210, df)          # Loyola Chicago vs Georgia Tech
    midwest_32_c = sim_winner(1397, 1333, df)          # Tennessee vs Oregon St
    midwest_32_d = sim_winner(1329, 1251, df)          # Oklahoma St vs Liberty
    midwest_32_e = sim_winner(1361, 1393, df)          # San Diego St vs Syracuse
    midwest_32_f = sim_winner(1452, 1287, df)          # West Virginia vs Morehead St
    midwest_32_g = sim_winner(1155, 1353, df)          # Clemson vs Rutgers
    midwest_32_h = sim_winner(1222, 1156, df)          # Houston vs Cleveland St
    print()
    print("-----Simulate Second Round-----")
    print("WEST")
    west_16_a = sim_winner(west_32_a, west_32_b, df)   
    west_16_b = sim_winner(west_32_c, west_32_d, df)  
    west_16_c = sim_winner(west_32_e, west_32_f, df)  
    west_16_d = sim_winner(west_32_g, west_32_h, df)           
    print("SOUTH")
    south_16_a = sim_winner(south_32_a, south_32_b, df)        
    south_16_b = sim_winner(south_32_c, south_32_d, df)       
    south_16_c = sim_winner(south_32_e, south_32_f, df)       
    south_16_d = sim_winner(south_32_g, south_32_h, df)  
    print("EAST")
    east_16_a = sim_winner(east_32_a, east_32_b, df) 
    east_16_b = sim_winner(east_32_c, east_32_d, df)       
    east_16_c = sim_winner(east_32_e, east_32_f, df)        
    east_16_d = sim_winner(east_32_g, east_32_h, df)        
    print("MIDWEST")
    midwest_16_a = sim_winner(midwest_32_a, midwest_32_b, df)   
    midwest_16_b = sim_winner(midwest_32_c, midwest_32_d, df)     
    midwest_16_c = sim_winner(midwest_32_e, midwest_32_f, df)   
    midwest_16_d = sim_winner(midwest_32_g, midwest_32_h, df) 
    print()
    print("-----Simulate Sweet Sixteen-----")
    print("WEST")
    west_8_a = sim_winner(west_16_a, west_16_b, df)   
    west_8_b = sim_winner(west_16_c, west_16_d, df)  
    print("SOUTH")
    south_8_a = sim_winner(south_16_a, south_16_b, df)        
    south_8_b = sim_winner(south_16_c, south_16_d, df)       
    print("EAST")
    east_8_a = sim_winner(east_16_a, east_16_b, df) 
    east_8_b = sim_winner(east_16_c, east_16_d, df)       
    print("MIDWEST")
    midwest_8_a = sim_winner(midwest_16_a, midwest_16_b, df)   
    midwest_8_b = sim_winner(midwest_16_c, midwest_16_d, df)     
    print()
    print("-----Simulate Elite Eight-----")
    print("WEST")
    west = sim_winner(west_8_a, west_8_b, df)   
    print("SOUTH")
    south = sim_winner(south_8_a, south_8_b, df)        
    print("EAST")
    east = sim_winner(east_8_a, east_8_b, df) 
    print("MIDWEST")
    midwest = sim_winner(midwest_8_a, midwest_8_b, df)   
    print()
    print("-----Simulate Semi Finals-----")
    print("WEST vs EAST")
    semi_a = sim_winner(west, east, df)   
    print("SOUTH vs MIDWEST")
    semi_b = sim_winner(south, midwest, df)        
    print()
    print("-----Simulate National Championship-----")
    champion = sim_winner(semi_a, semi_b, df)        
    print(sample_test(semi_a, semi_b, bst, 2021, feats, from_indy=True))



feats = ["adj_em", "luck", "sos_em", "ncsos_em", "away_win_pct", "home", "massey_rank"]
train, valid, test = split_data(df)
#train, valid, test = scale_features(train, valid, test, features)
bst, _, _ = train_model(train, valid, test, feats)
model = train_classifier(train, valid, test, feats)
#sample_test(1438, 1437, model, 2021, feature_cols=None, from_indy=True)

results = generateSample(feats, from_indy=True)
simulation(results)

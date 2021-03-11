import pandas as pd 
import numpy as np 
import os 
from string import digits
from scipy import stats
from sklearn import preprocessing

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")
data = {}
# LOAD CSV 
for f in os.listdir(data_dir):
    if f.endswith(".csv"):
        fname = f[:-4].lower()
        df_name = fname + '_df'
        data[df_name] = pd.read_csv(os.path.join(data_dir, f))
    else:
        continue
kenpom_df = pd.read_csv(os.path.join(data_dir, "kenpom.csv"))
# Assign Team ID to kenpom data
kenpom_df["name"] = kenpom_df["name"].str.lower()
kenpom_df["name"] = kenpom_df["name"].str.replace("*", "", regex=True)
kenpom_df["name"] = kenpom_df["name"].str.translate({ord(k): None for k in digits})
kenpom_df["name"] = kenpom_df["name"].str.strip()
kenpom_df = pd.merge(kenpom_df, data["mteamspellings_df"], how="left", left_on="name", right_on="TeamNameSpelling")

# assign Home city ID to team ID
distances_df = pd.read_csv("cities_w_dist.csv")
game_cities_df = data["mgamecities_df"][(data["mgamecities_df"]["CRType"] == "Regular")]
game_distances_df = pd.merge(game_cities_df, distances_df, how='left', on="CityID")
home_win_games_df = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WLoc"] == "H")]
team_homes_df = pd.merge(game_distances_df, home_win_games_df, how='left', on=["Season", "WTeamID"])
team_homes_df = team_homes_df[["WTeamID", "CityID", "City", "dist_to_indy"]].drop_duplicates(subset=["WTeamID"], keep='last')
max_dist = max(team_homes_df["dist_to_indy"])
min_dist = min(team_homes_df["dist_to_indy"])
team_homes_df["scale"] = (team_homes_df["dist_to_indy"] - min_dist) / (max_dist - min_dist)
team_homes_df["home_proxy"] = (team_homes_df["scale"] - 0.5) * -2

# Massey Ordinals

# remove leaked data from tourney compact results
def concat_row(r):
    if r['WTeamID'] < r['LTeamID']:
        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])
    else:
        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])
    return res

# Delete leaked from train
def delete_leaked_from_df_train(df_train, df_test):
    df_train['Concats'] = df_train.apply(concat_row, axis=1)
    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]
    df_train_idx = df_train_duplicates.index.values
    df_train = df_train.drop(df_train_idx)
    df_train = df_train.drop('Concats', axis=1)
    
    return df_train 
print("Tourney dataset size before")
print(data['mncaatourneycompactresults_df'].shape)
df_test = pd.read_csv(os.path.join(data_dir, "MSampleSubmissionStage1.csv"))
data['mncaatourneycompactresults_df'] = delete_leaked_from_df_train(data['mncaatourneycompactresults_df'], df_test)
print("Tourney dataset size after")
print(data['mncaatourneycompactresults_df'].shape)

# List of dfs, for reference
# To access a DF in this dictionary of DF use format: data["df_name"]
df_ls = []
for df in data:
    df_ls.append(df)
print(df_ls)

# HELPER FUNCTIONS
def getTeamName(team_id):
    """ returns TeamName from mteams_df given a TeamID """
    return data["mteams_df"][(data["mteams_df"]["TeamID"] == team_id)].values[0][1]

def checkPower6Conference(team_id):
    """ returns 1 if TeamID is in the Power 6 conferences """
    power_six = ['acc', 'big_ten', 'big_twelve', 'big_east', 'pac_twelve', 'sec']
    team_pd = data["mteamconferences_df"][(data["mteamconferences_df"]["Season"] == 2021) & (data["mteamconferences_df"]["TeamID"] == team_id)]
    if (len(team_pd) == 0):
        return 0
    confName = team_pd.iloc[0]['ConfAbbrev']
    return int(confName in power_six)

def getLastNChamps(team_id, year, n_years=35):
    """ returns the number of championships won in the past n_years (default=35) for a TeamID """
    champions_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 154)]
    champions_ls = champions_df[(champions_df["Season"] >= year - n_years)]["WTeamID"].tolist()
    return champions_ls.count(team_id)

def getLastNFinalFour(team_id, year, n_years=35):
    """ returns the number of Final Four appearances in the past n_years (default=35) for a TeamID """
    ffour_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 152)]
    ffour_ls = ffour_df[(ffour_df["Season"] >= year - n_years)]["WTeamID"].tolist()
    return ffour_ls.count(team_id)

def getLastNEliteEight(team_id, year, n_years=35):
    """ returns the number of Elite Eight appearances in the past n_years (default=35) for a TeamID """
    eeight_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 145) | (data["mncaatourneycompactresults_df"]["DayNum"] == 146)]
    eeight_ls = eeight_df[(eeight_df["Season"] >= year - n_years)]["WTeamID"].tolist()
    return eeight_ls.count(team_id)

def getNumWins(team_id, year, n_years=35):
    """ returns the number of wins in the past n_years for a TeamID """
    wins_df = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    return len(wins_df)

def getNumLosses(team_id, year, n_years=35):
    """ returns the number of losses in the past n_years for a TeamID """
    loss_df = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["LTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    return len(loss_df)

def getWinPct(team_id, year, n_years=35):
    """ returns total win percentage """
    wins = getNumWins(team_id, year, n_years)
    losses = getNumLosses(team_id, year, n_years)
    if wins + losses == 0:
        return np.nan
    else:
        return wins / (wins + losses)

def getHomeWinPct(team_id, year, n_years=35):
    """ returns home win percentage """
    wins = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    home_wins = len(wins[(wins["WLoc"] == "H")])
    losses = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["LTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    home_losses = len(losses[(losses["WLoc"] != "H")])
    if home_wins + home_losses == 0:
        return np.nan
    else:
        return home_wins / (home_wins + home_losses)

def getAwayWinPct(team_id, year, n_years=35):
    """ returns away win percentage """
    wins = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    losses = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["LTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    away_wins = len(wins[(wins["WLoc"] != "H")])
    away_losses = len(losses[(losses["WLoc"] == "H")])
    if away_wins + away_losses == 0:
        return np.nan
    else:
        return away_wins / (away_wins + away_losses)

def getHomeStat(row):
    if (row == 'H'):
        home = 1
    elif (row == 'A'):
        home = -1
    else:
        home = 0
    return home

def normalize(array):
    return stats.zscore(array, axis=1, nan_policy='omit')

## We can add other helper functions here (coach record, distance from home)

def getTeamSeasonStats(team_id, year):
    """
    Gathers 1 season's data from dfs for a given team_id. 
    Returns a vector of attributes
    """
    kenpom = kenpom_df[(kenpom_df["year"] == year) & (kenpom_df["TeamID"] == team_id)]
    team_home = team_homes_df[(team_homes_df["WTeamID"] == team_id)]
    massey = data['mmasseyordinals_df'][(data['mmasseyordinals_df']["Season"] == year) & (data['mmasseyordinals_df']["TeamID"] == team_id)]

    if len(kenpom) >= 1:
        rank = kenpom["rank"].values[0]
        seed = kenpom["seed"].values[0]
        adj_em = kenpom["adjem"].values[0]
        adj_o = kenpom["adjo"].values[0]
        adj_d = kenpom["adjd"].values[0]
        adj_t = kenpom["adjt"].values[0]
        luck = kenpom["luck"].values[0]
        SOS_EM = kenpom["SOS_EM"].values[0]
        SOS_O = kenpom["SOS_O"].values[0]
        SOS_D = kenpom["SOS_D"].values[0]
        NCSOS_EM = kenpom["NCSOS_EM"].values[0]
        wins = kenpom["wins"].values[0]
        losses = kenpom["losses"].values[0]
        win_pct = kenpom["win_pct"].values[0]
    else:
        rank = np.nan
        seed = np.nan
        adj_em = np.nan
        adj_o = np.nan
        adj_d = np.nan
        adj_t = np.nan
        luck = np.nan
        SOS_EM = np.nan
        SOS_O = np.nan
        SOS_D = np.nan
        NCSOS_EM = np.nan
    wins = getNumWins(team_id, year, 1)
    losses = getNumLosses(team_id, year, 1)
    win_pct = getWinPct(team_id, year, 1)
    home_win_pct = getHomeWinPct(team_id, year, 1)
    away_win_pct = getAwayWinPct(team_id, year, 1)
    power_six = checkPower6Conference(team_id)
    n_champs = getLastNChamps(team_id, year, 5)
    n_ffour = getLastNFinalFour(team_id, year, 5)
    n_eeight = getLastNEliteEight(team_id, year, 5)
    if len(team_home) < 1:
        home_proxy = np.nan
    else:
        home_proxy = team_home["home_proxy"].values[0]
    if len(massey) < 1:
        massey_rank = np.nan 
    else:
        massey_rank = massey["OrdinalRank"].values[0]

    return [rank, seed, adj_em, adj_o, adj_d, adj_t, luck, SOS_EM, SOS_O, SOS_D
            , NCSOS_EM, wins, losses, win_pct, home_win_pct, away_win_pct, power_six
            , n_champs, n_ffour, n_eeight, home_proxy, massey_rank]

def getSeasonStats(year):
    """
    Returns season statistics for every team in a given year
    """
    season_dict = {}
    for team in data["mteams_df"]["TeamID"]:
        team_stats = getTeamSeasonStats(team, year)
        season_dict[team] = team_stats
    return season_dict

def generateTrainingData(years):
    col_ls = ["rank", "seed", "adj_em", "adj_o", "adj_d", "adj_t"
                , "luck", "sos_em", "sos_o", "sos_d", "ncsos_em", "wins"
                , "losses", "win_pct", "home_win_pct", "away_win_pct", "power_six"
                , "n_champs", "n_ffour", "n_eeight", "home_proxy", "massey_rank", "home", "season", "WTeamID", "LTeamID", "result"]
    rows_list = []
    for year in years:
        print("Building year:", year)
        team_vectors = getSeasonStats(year)
        season = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["Season"] == year)]
        tourney = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["Season"] == year)]
        counter = 0
        for _, row in season.iterrows():
            w_team, l_team = row["WTeamID"], row["LTeamID"]
            w_vector = team_vectors[w_team]
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = getHomeStat(row["WLoc"])
            d = {}
            if counter % 2 == 0:
                diff.extend([home, year, w_team, l_team, 1])
                d = dict(zip(col_ls, diff))
                rows_list.append(d)
            else:
                negative_diff = [-x for x in diff]
                negative_diff.extend([home, year, w_team, l_team, 0])
                d = dict(zip(col_ls, negative_diff))
                rows_list.append(d)
            counter += 1
        for _, row in tourney.iterrows():
            w_team, l_team = row["WTeamID"], row["LTeamID"]
            w_vector = team_vectors[w_team]
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = 0
            if counter % 2 == 0:
                diff.extend([home, year, w_team, l_team, 1])
                d = dict(zip(col_ls, diff))
                rows_list.append(d)
            else:
                negative_diff = [-x for x in diff]
                negative_diff.extend([home, year, w_team, l_team, 0])
                d = dict(zip(col_ls, negative_diff))
                rows_list.append(d)
            counter += 1

    train = pd.DataFrame(rows_list)
    train.to_csv("training_data.csv", index=False)

    return train

def generateTeamVectors(years):
    cols = ["rank", "seed", "adj_em", "adj_o", "adj_d", "adj_t", "luck", "sos_em", "sos_o", "sos_d"
            , "ncsos_em", "wins", "losses", "win_pct", "home_win_pct", "away_win_pct", "power_six"
            , "n_champs", "n_ffour", "n_eeight", "home_proxy", "massey_rank", "home", "season", "teamID"]
    season_ls = []
    for year in years:
        team_vectors = getSeasonStats(year)
        for i in team_vectors:
            team_vectors[i].extend([0, year, i])
            d = dict(zip(cols, team_vectors[i]))
            season_ls.append(d)

    season_vectors = pd.DataFrame(season_ls)
    season_vectors.to_csv("team_vectors.csv", index=False)

    return season_vectors

year_range = range(2012, 2022)

print(generateTrainingData(year_range))   
print(generateTeamVectors(year_range))





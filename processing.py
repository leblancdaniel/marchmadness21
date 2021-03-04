import pandas as pd 
import numpy as np 
import os 
from string import digits

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
kenpom_df["name"] = kenpom_df["name"].str.replace("*", "")
kenpom_df["name"] = kenpom_df["name"].str.translate({ord(k): None for k in digits})
kenpom_df["name"] = kenpom_df["name"].str.strip()
kenpom_df = pd.merge(kenpom_df, data["mteamspellings_df"], how="left", left_on="name", right_on="TeamNameSpelling")
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
    return wins / (wins + losses)

def getHomeWinPct(team_id, year, n_years=35):
    """ returns home win percentage """
    wins = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    home_wins = len(wins[(wins["WLoc"] == "H")])
    losses = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["LTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    home_losses = len(losses[(losses["WLoc"] != "H")])

    return home_wins / (home_wins + home_losses)

def getAwayWinPct(team_id, year, n_years=35):
    """ returns away win percentage """
    wins = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["WTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    losses = data["mregularseasoncompactresults_df"][(data["mregularseasoncompactresults_df"]["LTeamID"] == team_id) & (data["mregularseasoncompactresults_df"]["Season"] > year - n_years)]
    away_wins = len(wins[(wins["WLoc"] != "H")])
    away_losses = len(losses[(losses["WLoc"] == "H")])
    return away_wins / (away_wins + away_losses)

## We can add other helper functions here (coach record, distance from home)

def getTeamSeasonStats(team_id, year):
    """
    Gathers 1 season's data from dfs for a given team_id. 
    Returns a vector of attributes
    """
    kenpom = kenpom_df[(kenpom_df["year"] == year) & (kenpom_df["TeamID"] == team_id)]
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
        home_win_pct = getHomeWinPct(team_id, year, 1)
        away_win_pct = getAwayWinPct(team_id, year, 1)
        power_six = checkPower6Conference(team_id)
        n_champs = getLastNChamps(team_id, year, 10)
        n_ffour = getLastNFinalFour(team_id, year, 10)
        n_eeight = getLastNEliteEight(team_id, year, 10)
        return [rank, seed, adj_em, adj_o, adj_d, adj_t, luck, SOS_EM, SOS_O, SOS_D
                , NCSOS_EM, wins, losses, win_pct, home_win_pct, away_win_pct, power_six
                , n_champs, n_eeight, n_ffour]
    else:
        pass
def getSeasonStats(year):
    """
    Returns season statistics for every team in a given year
    """
    season_dict = {}
    for team in data["mteams_df"]["TeamID"]:
        team_stats = getTeamSeasonStats(team, year)
        if team_stats is None:
            continue
        season_dict[team] = team_stats
    return season_dict

print(getSeasonStats(2019))




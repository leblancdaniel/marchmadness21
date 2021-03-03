import pandas as pd 
import numpy as np 
import os 


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

def getLastNChamps(team_id, n_years=35):
    """ returns the number of championships won in the past n_years (default=35) for a TeamID """
    champions_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 154)]
    champions_ls = champions_df[(champions_df["Season"] >= 2020 - n_years)]["WTeamID"].tolist()
    return champions_ls.count(team_id)

def getLastNFinalFour(team_id, n_years=35):
    """ returns the number of championships won in the past n_years (default=35) for a TeamID """
    ffour_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 152)]
    print(ffour_df)
    ffour_ls = ffour_df[(ffour_df["Season"] >= 2020 - n_years)]["WTeamID"].tolist()
    print(ffour_ls)
    return ffour_ls.count(team_id)

def getLastNEliteEight(team_id, n_years=35):
    """ returns the number of championships won in the past n_years (default=35) for a TeamID """
    eeight_df = data["mncaatourneycompactresults_df"][(data["mncaatourneycompactresults_df"]["DayNum"] == 145) | (data["mncaatourneycompactresults_df"]["DayNum"] == 146)]
    print(eeight_df)
    eeight_ls = eeight_df[(eeight_df["Season"] >= 2020 - n_years)]["WTeamID"].tolist()
    print(eeight_ls)
    return eeight_ls.count(team_id)


print(getLastNEliteEight(1437, 5))

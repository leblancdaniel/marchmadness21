import warnings, os
import numpy as np
import pandas as pd
import time 
import datetime
import re

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from scipy import stats

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")

def get_url(url):
    """ 
    Gets content of url by making GET request.  Returns HTML/XML text content, otherwise None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content 
            else:
                return None
    except RequestException as e:
        log_error('Error during requests to {} : {}'.format(url, str(e)))
        return None 

def is_good_response(resp):
    """ Returns True if response is HTML """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    print(e)


url = "https://kenpom.com/index.php?y="
lvars = 21
years = range(2002, 2022)
d = {}
for y in years:
    nurl = url + str(y) if y != 2021 else url[:-3]
    raw_html = get_url(nurl)
    soup = BeautifulSoup(raw_html)
    data = [s.text for s in soup.select("td")]
    ldata = len(data)
    d[y] = pd.DataFrame(np.array(data).reshape(int((ldata/lvars)), lvars))

years = list(d.keys())
start = years[-1]
yearsub = years[:-1]
kenpom = d[start]
kenpom['year'] = start 
for y in reversed(yearsub):
    work = d[y]
    work['year'] = y 
    kenpom = pd.concat([kenpom, work])

kenpom.columns = ["rank", "school", "conf", "w_l", "adjem", "adjo", "adjo_r",
                 "adjd", "adjd_r", "adjt", "adjt_r", "luck", "luck_r", "SOS_EM",
                 "SOS_EM_r", "SOS_O", "SOS_O_r", "SOS_D", "SOS_D_r", "NCSOS_EM", "NCSOS_EM_r", "year"]

meta = [["rank", 'int'], ["adjem", 'float'], ["adjo", 'float'], ["adjo_r", 'int'],
        ["adjd", 'float'], ["adjd_r", 'int'], ["adjt", 'float'], ["adjt_r", 'int'], 
        ["luck", 'float'], ["luck_r", 'int'], ["SOS_EM", 'float'], ["SOS_EM_r", 'int'], 
        ["SOS_O", 'float'], ["SOS_O_r", 'int'], ["SOS_D", 'float'], ["SOS_D_r", 'int'], 
        ["NCSOS_EM", 'float'], ["NCSOS_EM_r", 'int'], ["year", 'int']]
for m in meta:
    kenpom[m[0]] = kenpom[m[0]].astype(m[1])
kenpom.sort_values(['year', 'rank'], ascending=True, inplace=True)
kenpom.reset_index(inplace=True, drop=True)

kenpom['seed'] = [int(s.split()[-1]) if s.split()[-1].isdigit() else np.nan for s in kenpom['school']]
kenpom['name'] = [s if pd.isna(kenpom["seed"][i]) else " ".join(s.split()[:-1]) for (i,s) in enumerate(kenpom["school"])]
kenpom['wins'] = [int(s.split("-")[0]) for s in kenpom["w_l"]]
kenpom['losses'] = [int(s.split("-")[1]) for s in kenpom["w_l"]]
kenpom['win_pct'] = kenpom["wins"] / (kenpom["wins"] + kenpom["losses"])
kenpom = kenpom.drop(['school', 'w_l'], axis=1)
print(kenpom.dtypes)
print(kenpom.head())

outfile = "kenpom.csv"
kenpom.to_csv(outfile, index=False)

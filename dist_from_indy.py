import pandas as pd 
import numpy as np
import requests
import json 
import os 


current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")
cities = pd.read_csv(os.path.join(data_dir, "Cities.csv"))
cities["citystate"] = cities["City"].str.cat(cities["State"], sep=", ")
cities["dist_to_indy"] = np.nan

# enter your api key here 
api_key ='YOUR API KEY HERE'
# url variable store url  
url ='https://maps.googleapis.com/maps/api/distancematrix/json?'

for i, row in cities.iterrows():
    origin = row["City"]
    dest = "Indianapolis, IN"
    # Get method of requests module 
    r = requests.get(url + 'origins=' + origin 
                    + '&destinations=' + dest 
                    + '&key= ' + api_key) 
    result = r.json() 
    print(result)
    if result['rows'][0]['elements'][0]['status'] == 'OK':
        cities.at[i, 'dist_to_indy'] = result['rows'][0]['elements'][0]['distance']['value']
    else:
        pass

print(cities)
cities.to_csv("cities_w_dist.csv", index=False)

# marchmadness21
A fun project attempting to make a perfect bracket for the [2021 Kaggle Competition](https://www.kaggle.com/c/ncaam-march-mania-2021/data)!!!

## Top models from 2019 competition
Each year, Kaggle hosts a competition to come up with the best bracket prediction method.  The leaderboard ranking are sorted in ascending order by `log loss`.  In the 2019 competition, the 1st place received a log loss of `0.41477`, 5th place log loss was `0.43148`, and 10th place log loss was `0.43759`.  You can see the full 2019 leaderboard [here](https://www.kaggle.com/c/mens-machine-learning-competition-2019/leaderboard).

Interestingly, the first place model admits to getting "lucky" by manually overriding 3 first round predictions.  While other top submissions committed to a more programmatic approach, and taking data from external sources for additional team stats.

## Data sources
This [link](https://www.kaggle.com/c/ncaam-march-mania-2021/data) shares all of the data available for the 2021 competition, we've pulled this data into our `/data` directory.  

We also scraped data from [kenpom.com](kenpom.com) (run `get_kenpom.py` and move the output file `kenpom.csv` into the `/data` directory).  Because of some naming differences of schools, we did a little extra work to ensure we could connect each schools data in the `MTeamSpellings.csv` file.

We added in the distance from the tournament location (Indianapolis) using the Google Places Distance Matrix API as a feature, too.  The thinking here is that even though the tournament is in a neutral location, if my fanbase is much closer to Indy than yours, more of my fans will show up and give me a pseudo-home-advantage.  Just run `dist_from_indy.py` once you have your own API key from Google.

In order to build a new training dataset - just run `processing.py`.

## Simple model
We built a simple LightGBM model that did pretty well at first shot (0.845 AUC; 0.481 log loss).  

## Feature engineering (and extra data sources)
List a few features that would be nice to add (efficiency, schedule difficulty, distance from home, coach ratings,)

## Complex model(s)
We should scale the data feeding into our LightGBM model and then grid search the parameters to see what parameters should be.

Next we should try reducing number of dimensions/adding random logic to the output.  Turned out only 5 variables had the most predictive power in our model (`luck`, `adj_em`, `sos_em`, `ncsos_em`, and `home`).

We included the inverse scaled distance from Indianapolis as an approximation for `home` at the tournament called `home_proxy`.

The best logloss score on the practice leaderboard was `0.4173` landing us in the top 100 without much effort.

## Compare results on 2021 bracket
Thought it would be fun to track how several different models do, 
so we created a bracket group on [ESPN](https://fantasy.espn.com/tournament-challenge-bracket/2021/en/group?groupID=3931063).

Our personal brackets:
- daniel's puny human guess:
- steve's puny human guess: 

Our model brackets:
- Baseline Model: 
- Distance from Home Proxy: 
- Distance from Home + Luck Boost: 
- Home Proxy w/ Added Uncertainty: 

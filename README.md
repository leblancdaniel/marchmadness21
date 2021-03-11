# marchmadness21
A fun project attempting to make a perfect bracket for the [2021 Kaggle Competition](https://www.kaggle.com/c/ncaam-march-mania-2021/data)!!!

Here's the inital plan...  
- ~~Gather a few benchmarks from leaderboards from 2 years ago~~
- ~~Gather data from 2019 Kaggle competition~~
- ~~Create REALLY simple model to predict win probability~~
- ~~Engineer a bunch of new interesting features and use another simple model~~
- ~~Using the new features, create a more complex model~~
- ~~Reduce dimensionality (PCA, etc.) and create another model~~
- ~~Test & compare results on 2019 data~~
- Collect 2021 data & import into our dataset
- Test old models on 2021 bracket
- Test new models on 2021 bracket

## Top models from 2019 competition
Each year, Kaggle hosts a competition to come up with the best bracket prediction method.  The leaderboard ranking are sorted in ascending order by `log loss`.  In the 2019 competition, the 1st place received a log loss of `0.41477`, 5th place log loss was `0.43148`, and 10th place log loss was `0.43759`.  You can see the full 2019 leaderboard [here](https://www.kaggle.com/c/mens-machine-learning-competition-2019/leaderboard).

Interestingly, the first place model admits to getting "lucky" by manually overriding 3 first round predictions.  While other top submissions committed to a more programmatic approach, and taking data from external sources for additional team stats.

## Data sources
This [link](https://www.kaggle.com/c/ncaam-march-mania-2021/data) shares all of the data available for the 2021 competition, we've pulled this data into our `/data` directory.  

We also scraped data from [kenpom.com](kenpom.com) and put that in the `/data` directory.  Because of some naming differences of schools, we did a little extra work to ensure we could connect each schools data in the `MTeamSpellings.csv` file.

We added in the distance from the tournament location (Indianapolis) using the Google Places Distance Matrix API as a feature, too.  The thinking here is that even though the tournament is in a neutral location, if my fanbase is much closer to Indy than yours, more of my fans will show up and give me a pseudo-home-advantage.

In order to build a new training dataset - just run `processing.py`.

## Simple model
We built a simple LightGBM model that did pretty well at first shot (0.845 AUC; 0.481 log loss).  

## Feature engineering (and extra data sources)
List a few features that would be nice to add (efficiency, schedule difficulty, distance from home, coach ratings,)

## Complex model(s)
We should scale the data feeding into our LightGBM model and then grid search the parameters to see what parameters should be.

Next we should try reducing number of dimensions/adding random logic to the output.  Turned out only 5 variables had the most predictive power in our model (`luck`, `adj_em`, `sos_em`, `ncsos_em`, and `home`).

## Import 2021 data


## Old model results on 2021 bracket


## Re-train models using 2021 data


## New model results on 2021 bracket



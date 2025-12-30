import pandas as pd
import pickle
import sys
import warnings

from tqdm import tqdm
from datetime import timedelta

warnings.filterwarnings("ignore")

rolling_window = 5

all_emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

posts_data = pd.read_csv("../data/fb_data_with_predictions.csv")
highlevel_unions = pd.read_csv("../data/mapping_fb_unions.csv")
highlevel_unions = highlevel_unions.dropna(subset=["account_match"])
highlevel_unions_handles = highlevel_unions[["union", "account_match"]]
highlevel_unions_handles.columns = ["main_union", "handle"]

posts_data = posts_data.merge(highlevel_unions_handles, how="left", left_on="surface.username", right_on="handle")

emotion_predictions = pd.read_csv("../data/posts_emotion_predictions_mean-pool.csv")

posts_data = posts_data.merge(emotion_predictions, on="id", how="inner")

full_date_range = pd.date_range(start="2015-01-01", end="2024-12-31")

all_results = []
for main_union in tqdm(posts_data["main_union"].unique(), desc="Processing main unions"):

    union_data = posts_data[posts_data["main_union"] == main_union].copy()

    union_data['creation_time'] = pd.to_datetime(union_data['creation_time'])
    union_data["date"] = union_data["creation_time"].dt.date

    # Aggregate by date → mean emotions
    daily = union_data.groupby("date")[all_emotions].mean()

    # Reindex to full date range
    daily = daily.reindex(full_date_range)

    # Fill missing days with zeros
    daily_filled = daily.fillna(0)

    # Rolling window of size 5
    rolled = daily_filled.rolling(window=rolling_window, center=True).mean()

    rolled["main_union"] = main_union
    rolled["date"] = rolled.index

    all_results.append(rolled.reset_index(drop=True))    

final_df = pd.concat(all_results, ignore_index=True)

final_df.to_csv("../data/rolling_emotions_by_union.csv", index=False)
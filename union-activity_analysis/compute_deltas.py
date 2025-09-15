import pandas as pd
import pickle
import sys
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")

rolling_window = int(sys.argv[1])
event = sys.argv[2]

def get_period(days):
    if -7 <= days <= -3:
        return "before"
    elif -2 <= days <= 2:
        return "during"
    elif 3 <= days <= 7:
        return "after"
    else:
        return "unknown"
    
posts_data = pd.read_csv("../data/fb_data_with_predictions.csv")
highlevel_unions = pd.read_csv("../data/mapping_fb_unions.csv")
highlevel_unions = highlevel_unions.dropna(subset=["account_match"])
highlevel_unions_handles = highlevel_unions[["union", "account_match"]]
highlevel_unions_handles.columns = ["main_union", "handle"]

posts_data = posts_data.merge(highlevel_unions_handles, how="left", left_on="surface.username", right_on="handle")


# Load events dictionary file
with open("../data/events_dict_fb_all2024_new.json", "r") as f:
    events_dict_complete = json.load(f)

all_cases = []
union_no_cases = 0
for union in events_dict_complete.keys():
    if union not in posts_data["main_union"].values:
        continue
    if len(events_dict_complete[union]["cases"]) == 0:
        union_no_cases += 1
    for case in events_dict_complete[union]["cases"]:
        if case["case_losing_election_date"] != "None" or case["case_winning_election_date"] != "None":
            all_cases.append(case["case_number"])

# Load actual proportions file
with open(f"../data/actual_proportions_{rolling_window}_{event}.pkl", "rb") as f:
    actual_proportions = pickle.load(f)

# Load baseline proportions file
baseline_proportions = pd.read_csv(f"../data/baseline_proportions_{rolling_window}_{event}.csv")

# Compute deltas, case by case
all_data = []
for main_union in tqdm(events_dict_complete.keys()):
    actual_proportions_union = actual_proportions[actual_proportions["main_union"] == main_union]
    baseline_proportions_union = baseline_proportions[baseline_proportions["main_union"] == main_union]

    if len(baseline_proportions_union) == 0:
        continue

    for case in events_dict_complete[main_union]["cases"]:
        case_number = case["case_number"]
        actual_case = actual_proportions_union.copy()
        baseline_proportion_case = baseline_proportions_union.copy()

        actual_case["case_number"] = case_number

        ## don't compute deltas for cases without election dates
        if event=="Losing Election":
            if case["case_losing_election_date"] == "None":
                continue
            else:
                event_date = case["case_losing_election_date"]
        elif event=="Winning Election":
            if case["case_winning_election_date"] == "None":
                continue
            else:
                event_date = case["case_winning_election_date"]

        # if event date is NaT, continue
        if pd.isna(pd.to_datetime(event_date, errors="coerce")):
            continue

        actual_case["event_date"] = pd.to_datetime(event_date)

        actual_case["days_from_event"] = (actual_case["event_date"] - actual_case["date"]).dt.days

        actual_case = actual_case[actual_case["days_from_event"].between(-7, 7, inclusive="both")]

        actual_case["period"] = actual_case["days_from_event"].apply(get_period)

        prop_cols = [c for c in actual_case.columns if c.endswith("_prop")]

        melted = actual_case.melt(
            id_vars=["main_union", "case_number", "period", "event_date"],
            value_vars=prop_cols,
            var_name="frame",
            value_name="prop"
        )

        agg = melted.groupby(["main_union", "case_number", "frame", "period", "event_date"])["prop"].mean().reset_index()

        # Pivot so before/during/after are columns
        agg_pivot = agg.pivot_table(
            index=["main_union", "case_number", "frame", "event_date"],
            columns="period",
            values="prop"
        ).reset_index()

        # Optional: rename columns for clarity
        agg_pivot.columns.name = None
        agg_pivot.rename(columns={"before": "before_avg", "during": "during_avg", "after": "after_avg", "frame": "frame_prop"}, inplace=True)

        df_merged = pd.merge(
            agg_pivot,
            baseline_proportion_case,
            on=["main_union", "frame_prop"],
            how="left",
            suffixes=("", "_baseline")
        )

        df_merged["delta_before"] = df_merged["before_avg"] - df_merged["before_avg_baseline"]
        df_merged["delta_during"] = df_merged["during_avg"] - df_merged["during_avg_baseline"]
        df_merged["delta_after"] = df_merged["after_avg"] - df_merged["after_avg_baseline"]

        all_data.append(df_merged)

df_all = pd.concat(all_data, ignore_index=True)

with open(f"../data/deltas_{rolling_window}_{event}.pkl", "wb") as f:
    pickle.dump(df_all, f)
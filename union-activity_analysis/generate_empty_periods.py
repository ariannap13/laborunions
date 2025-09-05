import pandas as pd
import pickle
import sys
import warnings
import json

warnings.filterwarnings("ignore")

# load events dict to json
with open("../data/events_dict_fb_all2024.json", "r") as f:
    events_dict_complete = json.load(f)

# filter on only case_losing_election_date and case_winning_election_date
desired_fields = ["case_number", "case_losing_election_date", "case_winning_election_date"]

    # Function to compute 18-day no-event periods
def compute_no_event_periods(dates_series):
    no_event_periods = []
    if not dates_series.empty:
        dates_set = set(dates_series.dt.date)
        current_start = None
        consecutive_days = 0
        for day in full_date_range:
            if day.date() not in dates_set:
                if current_start is None:
                    current_start = day
                consecutive_days += 1
            else:
                if consecutive_days >= 18:
                    start = current_start
                    while consecutive_days >= 18:
                        end = start + pd.Timedelta(days=17)
                        if end > full_date_range[-1]:
                            end = full_date_range[-1]
                        no_event_periods.append((start.date(), end.date()))
                        start += pd.Timedelta(days=18)
                        consecutive_days -= 18
                current_start = None
                consecutive_days = 0
        # Handle stretch till end
        if consecutive_days >= 18:
            start = current_start
            while consecutive_days >= 18:
                end = start + pd.Timedelta(days=17)
                if end > full_date_range[-1]:
                    end = full_date_range[-1]
                no_event_periods.append((start.date(), end.date()))
                start += pd.Timedelta(days=18)
                consecutive_days -= 18
    return no_event_periods

dict_complete = {}
for main_union in events_dict_complete.keys():
    dict_complete[main_union] = []
    for case in events_dict_complete[main_union]["cases"]:
        filtered_case = {field: case.get(field, None) for field in desired_fields}
        dict_complete[main_union].append(filtered_case)

    # Collect all election dates and convert to datetime
    losing_dates = pd.to_datetime([
        case["case_losing_election_date"] 
        for case in dict_complete[main_union] 
        if case.get("case_losing_election_date") != "None"
    ])
    winning_dates = pd.to_datetime([
        case["case_winning_election_date"] 
        for case in dict_complete[main_union] 
        if case.get("case_winning_election_date") != "None"
    ])

    # Convert to Series before concatenating
    all_dates = pd.Series(losing_dates.tolist() + winning_dates.tolist()) if len(losing_dates) + len(winning_dates) > 0 else pd.Series(dtype='datetime64[ns]')
    dates_losing = pd.Series(losing_dates.tolist()) if len(losing_dates) > 0 else pd.Series(dtype='datetime64[ns]')
    dates_winning = pd.Series(winning_dates.tolist()) if len(winning_dates) > 0 else pd.Series(dtype='datetime64[ns]')

    # Full date range
    full_date_range = pd.date_range(start="2015-01-01", end="2024-12-31", freq='D')
    
    # Compute no-event periods
    dict_complete[main_union + "_no_event_periods"] = compute_no_event_periods(all_dates)
    dict_complete[main_union + "_no_event_losing"] = compute_no_event_periods(dates_losing)
    dict_complete[main_union + "_no_event_winning"] = compute_no_event_periods(dates_winning)

# Separate dictionaries for the three types of no-event periods
dict_no_event_all = {k.replace("_no_event_periods", ""): v for k, v in dict_complete.items() if "_no_event_periods" in k}
dict_no_event_losing = {k.replace("_no_event_losing", ""): v for k, v in dict_complete.items() if "_no_event_losing" in k}
dict_no_event_winning = {k.replace("_no_event_winning", ""): v for k, v in dict_complete.items() if "_no_event_winning" in k}

# Save each as a separate pickle file
with open("../data/empty_periods_all.pkl", "wb") as f:
    pickle.dump(dict_no_event_all, f)

with open("../data/empty_periods_all_Losing Election.pkl", "wb") as f:
    pickle.dump(dict_no_event_losing, f)

with open("../data/empty_periods_all_Winning Election.pkl", "wb") as f:
    pickle.dump(dict_no_event_winning, f)
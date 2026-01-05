import pandas as pd
import pickle
import sys
import warnings
import json

print("all imported")

warnings.filterwarnings("ignore")

diag_prog_single = True

balance_win_loss = True # decide whether to balance the number of wins and losses per union
subsample_overly_represented = False # decide whether to subsample overly active unions

seed = int(sys.argv[1])
rolling_window = int(sys.argv[2])

def get_period_rolling(days):
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

# Load actual proportions file - working data file
if diag_prog_single == True:
    with open(f"../data/actual_proportions_{rolling_window}_Winning Election_diag+prog.pkl", "rb") as f:
        actual_proportions = pickle.load(f)
else:
    with open(f"../data/actual_proportions_{rolling_window}_Winning Election.pkl", "rb") as f:
        actual_proportions = pickle.load(f)

# Load events dictionary file - working data file
with open("../data/events_dict_fb_complete_all_contracts_all2024_new.json", "r") as f:
    events_dict_complete = json.load(f)

rolling_emotions = pd.read_csv("../data/rolling_emotions_by_union.csv") # working file

all_cases = []
union_no_cases = 0
for union in events_dict_complete.keys():
    actual_proportions_union = actual_proportions[actual_proportions["main_union"] == union]
    rolling_emotions_union = rolling_emotions[rolling_emotions["main_union"] == union]
    if union not in posts_data["main_union"].values:
        continue
    if len(events_dict_complete[union]["cases"]) == 0:
        union_no_cases += 1
    for case in events_dict_complete[union]["cases"]:
        if case["case_losing_election_date"] != "None" or case["case_winning_election_date"] != "None":
            election_id = case["case_number"]
            election_state = case["case_state"]

            actual_case = actual_proportions_union.copy()
            actual_case["case_number"] = election_id

            emotions_case = rolling_emotions_union.copy()

            election_win = 1 if case["case_winning_election_date"] != "None" else 0
            election_date = case["case_winning_election_date"] if election_win == 1 else case["case_losing_election_date"]
            posts_union = posts_data[posts_data["main_union"] == union]
            election_date = pd.to_datetime(election_date).tz_localize(None)
            posts_union["creation_time"] = pd.to_datetime(posts_union["creation_time"]).dt.tz_localize(None)
            posts_union_before = posts_union[posts_union["creation_time"] < election_date]
            # compute the number of posts per week
            posts_union_weekly = posts_union.set_index("creation_time").groupby(pd.Grouper(freq="W")).size()

            actual_case["event_date"] = pd.to_datetime(election_date)
            actual_case["days_from_event"] = (actual_case["event_date"] - actual_case["date"]).dt.days
            actual_case = actual_case[actual_case["days_from_event"].between(-7, 7, inclusive="both")]
            actual_case["period"] = actual_case["days_from_event"].apply(get_period_rolling)

            case_before = actual_case[actual_case["period"]=="before"]

            if len(case_before) == 0:
                if diag_prog_single:
                    diagprog_prop = 0
                else:
                    diag_prop = 0
                    prog_prop = 0
                mot_prop = 0
                comm_prop = 0
                eng_prop = 0
            else:
                if diag_prog_single:
                    diagprog_prop = case_before["rolling_diagnostic+prognotic_prop"].mean()
                else:
                    diag_prop = case_before["rolling_diagnostic_prop"].mean()
                    prog_prop = case_before["rolling_prognotic_prop"].mean()
                mot_prop = case_before["rolling_motivational_prop"].mean()
                comm_prop = case_before["rolling_community_prop"].mean()
                eng_prop = case_before["rolling_engagement_prop"].mean()

            emotions_case["event_date"] = pd.to_datetime(election_date)
            emotions_case["date"] = pd.to_datetime(emotions_case["date"])
            emotions_case["days_from_event"] = (emotions_case["event_date"] - emotions_case["date"]).dt.days
            emotions_case = emotions_case[emotions_case["days_from_event"].between(-7, 7, inclusive="both")]
            emotions_case["period"] = emotions_case["days_from_event"].apply(get_period_rolling)

            emotions_before = emotions_case[emotions_case["period"]=="before"]

            if len(emotions_before) == 0:
                anger_prob = 0
                disgust_prob = 0
                fear_prob = 0
                joy_prob = 0
                neutral_prob = 0
                sadness_prob = 0
                surprise_prob = 0
            else:
                anger_prob = emotions_before["anger"].mean()
                disgust_prob = emotions_before["disgust"].mean()
                fear_prob = emotions_before["fear"].mean()
                joy_prob = emotions_before["joy"].mean()
                neutral_prob = emotions_before["neutral"].mean()
                sadness_prob = emotions_before["sadness"].mean()
                surprise_prob = emotions_before["surprise"].mean()

            if diag_prog_single:
                all_cases.append({
                    "union": union,
                    "total_posts_union": len(posts_union),
                    "total_posts_before": len(posts_union_before),
                    "weekly_posts_mean": posts_union_weekly.mean(),
                    "election_id": election_id,
                    "election_state": election_state,
                    "election_win": election_win,
                    "diag+prog_prop": diagprog_prop,
                    "mot_prop": mot_prop,
                    "comm_prop": comm_prop,
                    "eng_prop": eng_prop,
                    "anger_prob": anger_prob,
                    "disgust_prob": disgust_prob,
                    "fear_prob": fear_prob,
                    "joy_prob": joy_prob,
                    "neutral_prob": neutral_prob,
                    "sadness_prob": sadness_prob,
                    "surprise_prob": surprise_prob
                    })
            else:
                all_cases.append({
                    "union": union,
                    "total_posts_union": len(posts_union),
                    "total_posts_before": len(posts_union_before),
                    "weekly_posts_mean": posts_union_weekly.mean(),
                    "election_id": election_id,
                    "election_state": election_state,
                    "election_win": election_win,
                    "diag_prop": diag_prop,
                    "prog_prop": prog_prop,
                    "mot_prop": mot_prop,
                    "comm_prop": comm_prop,
                    "eng_prop": eng_prop,
                    "anger_prob": anger_prob,
                    "disgust_prob": disgust_prob,
                    "fear_prob": fear_prob,
                    "joy_prob": joy_prob,
                    "neutral_prob": neutral_prob,
                    "sadness_prob": sadness_prob,
                    "surprise_prob": surprise_prob
                    })

df = pd.DataFrame(all_cases)

# Get additional information
union_dict_complete = {
'afscme': ['Public Sector', 'Industrial'],
'afge': ['Public Sector', 'Industrial'],
'csea': ['Public Sector', 'Industrial'],
'iaff': ['Public Sector', 'Craft'],
'aft': ['Education', 'Industrial'],
'nea': ['Education', 'Industrial'],
'nnu': ['Health Care', 'Industrial'],
'seiu': ['Health Care', 'Industrial'],
'ibew': ['Building Trades / Construction', 'Craft'],
'iupat': ['Building Trades / Construction', 'Craft'],
'ironworkers': ['Building Trades / Construction', 'Craft'],
'bac': ['Building Trades / Construction', 'Craft'],
'ua': ['Building Trades / Construction', 'Craft'],
'roofers and waterproofers': ['Building Trades / Construction', 'Craft'],
'liuna': ['Building Trades / Construction', 'Craft'],
'smart': ['Building Trades / Construction', 'Craft'],
'ibb': ['Building Trades / Construction', 'Craft'],
'natca': ['Transportation', 'Craft'],
'twu': ['Transportation', 'Industrial'],
'atu': ['Transportation', 'Industrial'],
'siu': ['Transportation', 'Industrial'],
'meba': ['Transportation', 'Craft'],
'uaw': ['Manufacturing & Industrial', 'Industrial'],
'usw': ['Manufacturing & Industrial', 'Industrial'],
'iam': ['Manufacturing & Industrial', 'Industrial'],
'umwa': ['Manufacturing & Industrial', 'Industrial'],
'bctgm': ['Manufacturing & Industrial', 'Industrial'],
'uwua': ['Manufacturing & Industrial', 'Industrial'],
'rwdsu': ['Service, Retail & Hospitality', 'Industrial'],
'ufcw': ['Service, Retail & Hospitality', 'Industrial'],
'unite here': ['Service, Retail & Hospitality', 'Industrial'],
'iatse': ['Entertainment & Media', 'Craft'],
'sag-aftra': ['Entertainment & Media', 'Craft'],
'afm': ['Entertainment & Media', 'Craft'],
'wgae': ['Entertainment & Media', 'Craft'],
'cwa': ['Entertainment & Media', 'Industrial'],
'opeiu': ['Entertainment & Media', 'Industrial'],
'ifpte': ['Other', 'Industrial'],
'apwu': ['Other', 'Industrial'],
'ibt': ['Other', 'Industrial']
}

df_union_info_type = pd.DataFrame.from_dict(union_dict_complete, orient='index', columns=['sector', 'type'])
df_union_info_type.index.name = 'union'
df_union_info_type.reset_index(inplace=True)
df_union_info_all = pd.read_excel('../data/union_info.xlsx')

df_union = df_union_info_all.merge(df_union_info_type, on='union', how='left')
df_union = df_union.merge(df, on="union", how="left")

# load right-to-work states
rightwork = pd.read_csv("../data/right-to-work_states.csv")
df_union["rightwork"] = df_union["election_state"].isin(rightwork["Abbreviation"]).astype(int)

if subsample_overly_represented:
    df_union_to_save = df_union.copy()
    upper_threshold = df_union_to_save.groupby('union').size().quantile(0.90)
    target_size = int(upper_threshold)
    # Subsample each union to its target size
    sampled = []
    for union, group in df_union_to_save.groupby("union"):
        target_n = target_size
        if len(group) > upper_threshold:
            sampled_group = group.sample(n=target_n, random_state=seed)
        else:
            sampled_group = group  # keep all if fewer than target
        sampled.append(sampled_group)
    df_sampled = pd.concat(sampled, ignore_index=True)
    df_union = df_sampled.copy()

if balance_win_loss:
    df_union_to_save = df_union.copy()
    for union in df_union["union"].unique():
        df_union_union = df_union[df_union["union"] == union]
        wins = df_union_union[df_union_union["election_win"] == 1]
        losses = df_union_union[df_union_union["election_win"] == 0]
        n_wins = len(wins)
        n_losses = len(losses)
        if n_wins > n_losses:
            wins_sampled = wins.sample(n=n_losses, random_state=seed)
            df_union_to_save = pd.concat([df_union_to_save[df_union_to_save["union"] != union], wins_sampled, losses], ignore_index=True)
        elif n_losses > n_wins:
            losses_sampled = losses.sample(n=n_wins, random_state=seed)
            df_union_to_save = pd.concat([df_union_to_save[df_union_to_save["union"] != union], wins, losses_sampled], ignore_index=True)
    df_union = df_union_to_save.copy()

# select tags for names:
if diag_prog_single:
    tag_diag_prog = "_diag+prog"
else:
    tag_diag_prog = ""

if balance_win_loss:
    tag_balance_winloss = "_balanced"
else:
    tag_balance_winloss = ""

if subsample_overly_represented:
    tag_subsample_overly = "_subsample-active"
else:
    tag_subsample_overly = ""

df_union.to_csv(f'../data/union_info_bayesian_rolling_window_{rolling_window}_seed{seed}{tag_diag_prog}{tag_balance_winloss}{tag_subsample_overly}-moredata.csv', index=False) # working file

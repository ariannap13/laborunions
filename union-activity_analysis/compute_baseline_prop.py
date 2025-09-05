import pandas as pd
import pickle
import sys
import warnings

from tqdm import tqdm
from datetime import timedelta

warnings.filterwarnings("ignore")

intervention_type = sys.argv[1]
rolling_window = int(sys.argv[2])
centered_tag = "_centered"
specific_empty_periods = True

all_frames = ["diagnostic", "prognostic", "motivational", "community", "engagement"] 

# Open online activity data
df_frames = pd.read_csv("../data/fb_data_with_predictions.csv", on_bad_lines='skip')

# Open random events file
if specific_empty_periods:
    with open(f"../data/empty_periods_all_{intervention_type}.pkl", "rb") as f:
        empty_periods_dict = pickle.load(f)
else:
    with open(f"../data/empty_periods_all.pkl", "rb") as f:
        empty_periods_dict = pickle.load(f)

## Add info on main union (based on surface.username)
highlevel_unions = pd.read_csv("../data/mapping_fb_unions.csv")
highlevel_unions = highlevel_unions.dropna(subset=["account_match"])
highlevel_unions_handles = highlevel_unions[["union", "account_match"]]
highlevel_unions_handles.columns = ["main_union", "surface.username"]

df_frames = df_frames.merge(highlevel_unions_handles, how="left", on="surface.username")

all_results_list_baseline = []
all_results_list_actual = []
for main_union in tqdm(df_frames["main_union"].unique(), desc="Processing main unions"):
    if main_union not in empty_periods_dict:
        continue
    union_data = df_frames[df_frames["main_union"] == main_union]

    union_data['creation_time'] = pd.to_datetime(union_data['creation_time'])
    union_data["date"] = union_data["creation_time"].dt.date

    # Step 1: create co-occurrence columns at post level
    for target_frame in all_frames:
        for frame_co in all_frames:
            if frame_co != target_frame:
                # this counts both for frame alone and for co-occurrence
                union_data[f'{frame_co}_with_{target_frame}'] = (
                    (union_data[frame_co] == 1) & (union_data[target_frame] == 1)
                ).astype(int)
            else:
                union_data[f'{frame_co}_with_{target_frame}'] = (
                    (union_data[target_frame] == 1) &
                    (union_data[[f for f in all_frames if f != target_frame]].sum(axis=1) == 0)
                ).astype(int)

    # Step 2: build aggregation dictionary including co-occurrence columns
    agg_dict = {frame: 'sum' for frame in all_frames}
    agg_dict.update({col: 'sum' for col in union_data.columns if '_with_' in col})
    agg_dict['id'] = 'count'

    union_data = union_data.groupby("date").agg(agg_dict).reset_index()

    # Step 3: fill in missing dates
    full_date_range = pd.date_range(start="2015-01-01", end="2024-12-31")
    union_data = union_data.set_index('date').reindex(full_date_range, fill_value=0).rename_axis('date').reset_index()

    # Step 4: set date index for rolling
    union_data.set_index('date', inplace=True)

    # Step 5: compute rolling totals for single frames
    union_data['rolling_total_videos'] = union_data['id'].rolling(window=rolling_window, center=True).sum()

    # Compute rolling totals for posts with any frame
    mask_anyframe = (union_data[all_frames].sum(axis=1) > 0).astype(int)

    union_data["any_frame"] = mask_anyframe

    # rolling sum but only over rows where any_frame == 1
    union_data["rolling_any_frame"] = (
        union_data["id"]
        .where(union_data["any_frame"] == 1, 0)   # keep id if any_frame else 0
        .rolling(window=rolling_window, center=True)
        .sum()
    )
    for frame in all_frames:
        # Rolling sum of posts with this frame
        union_data[f'rolling_{frame}_frames'] = union_data[frame].rolling(window=rolling_window, center=True).sum()
        union_data[f'rolling_{frame}_prop'] = union_data[f'rolling_{frame}_frames'] / union_data['rolling_total_videos']

    # Step 6: compute rolling proportions for co-occurrence
    for target_frame in all_frames:
        rolling_target = union_data[target_frame].rolling(window=rolling_window, center=True).sum()

        for frame_co in all_frames:

            colname = f'{frame_co}_with_{target_frame}'
            rolling_col = f'rolling_{colname}_frames'
            prop_col = f'rolling_{colname}_prop'

            # Rolling sum of co-occurrence counts
            union_data[rolling_col] = union_data[colname].rolling(window=rolling_window, center=True).sum()

            # Proportion relative to target frame
            union_data[prop_col] = union_data[rolling_col] / rolling_target

    # Identify rolling proportion columns
    rolling_prop_cols = [col for col in union_data.columns if col.endswith('_prop')]
    union_data["main_union"] = main_union
    # We want to fill NaN with 0
    union_data.fillna(0, inplace=True)

    all_results_list_actual.append(union_data[rolling_prop_cols + ['main_union', 'rolling_any_frame', 'rolling_total_videos']].reset_index())

    # transform union_data.index to datetime.date
    union_data.index = union_data.index.date

    # Step 7: compute the before, during and after percentage for each baseline event
    all_possible_empty_periods = empty_periods_dict[main_union]

    results = []  # to store results per event

    for empty_period in all_possible_empty_periods:
        baseline_event = empty_period[0] + timedelta(days=(empty_period[1] - empty_period[0]).days // 2)
        # transform to datetime.date
        
        # Days from intervention
        union_data["days_from_intervention"] = (union_data.index - baseline_event).days

        # Select windows
        before_mask = union_data["days_from_intervention"].between(-7, -3, inclusive="both")
        during_mask = union_data["days_from_intervention"].between(-2, 2, inclusive="both")
        after_mask  = union_data["days_from_intervention"].between(3, 7, inclusive="both")

        for col in rolling_prop_cols:
            before_avg = union_data.loc[before_mask, col].mean()
            during_avg = union_data.loc[during_mask, col].mean()
            after_avg  = union_data.loc[after_mask, col].mean()

            results.append({
                "main_union": main_union,
                "baseline_event": baseline_event,
                "frame_prop": col,
                "before_avg": before_avg,
                "during_avg": during_avg,
                "after_avg": after_avg
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    all_results_list_baseline.append(results_df)

results_all_baseline = pd.concat(all_results_list_baseline, ignore_index=True)

# Step 8: aggregate mean by main_union and frame_prop
results_agg = results_all_baseline.groupby(["main_union", "frame_prop"]).mean().reset_index()

# Step 9: save results to CSV
results_agg.to_csv(f"../data/baseline_proportions_{rolling_window}_{intervention_type}.csv", index=False)

results_all_actual = pd.concat(all_results_list_actual, ignore_index=True)
with open(f"../data/actual_proportions_{rolling_window}_{intervention_type}.pkl", "wb") as f:
    pickle.dump(results_all_actual, f)
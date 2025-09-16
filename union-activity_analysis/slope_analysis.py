import pandas as pd
import pickle
from tqdm import tqdm

treatment = "remove_over"  # choose between "remove_over" or ""

def subsample_unions(df_losing, df_winning, random_state=0):
    # Combine both datasets
    df_all = pd.concat([df_losing, df_winning], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["main_union", "case_number"])
    
    # Count cases per union
    size_counts = df_all.groupby("main_union").size()

    # Define upper bounds (90th percentile)
    upper = size_counts.quantile(0.90)
    
    # Decide target sample size (here: clip to [lower, upper])
    target_counts = size_counts.clip(upper=upper).astype(int)
    
    # Subsample each union to its target size
    sampled = []
    for union, group in df_all.groupby("main_union"):
        target_n = target_counts.loc[union]
        if len(group) > upper:
            sampled_group = group.sample(n=target_n, random_state=random_state)
        else:
            sampled_group = group  # keep all if fewer than target
        sampled.append(sampled_group)
    df_sampled = pd.concat(sampled, ignore_index=True)
    
    # Re-split into losing and winning
    df_sub_losing = df_losing[df_losing["case_number"].isin(df_sampled["case_number"])]
    df_sub_winning = df_winning[df_winning["case_number"].isin(df_sampled["case_number"])]
    
    return df_sub_losing, df_sub_winning

with open(f"../data/deltas_5_Losing Election.pkl", "rb") as f:
    losing_deltas = pickle.load(f)

with open(f"../data/deltas_5_Winning Election.pkl", "rb") as f:
    winning_deltas = pickle.load(f)

losing_deltas["delta_diff"] = (losing_deltas["delta_after"] - losing_deltas["delta_before"])
winning_deltas["delta_diff"] = (winning_deltas["delta_after"] - winning_deltas["delta_before"])

losing_deltas = losing_deltas[~losing_deltas['frame_prop'].str.contains("with")]
winning_deltas = winning_deltas[~winning_deltas['frame_prop'].str.contains("with")]

# Remove over-represented unions
if treatment == "remove_over":
    for seed in tqdm(range(20)):

        losing_deltas, winning_deltas = subsample_unions(losing_deltas, winning_deltas, seed)

        # overall
        overall_deltas = pd.concat([losing_deltas, winning_deltas], ignore_index=True)

        # Get value of 25% and 75% quantiles
        q1 = overall_deltas["delta_diff"].quantile(0.25)
        q3 = overall_deltas["delta_diff"].quantile(0.75)

        losing_deltas["delta_diff_category"] = pd.cut(
            losing_deltas["delta_diff"],
            bins=[-float("inf"), q1, q3, float("inf")],
            labels=["decrease", "stable", "increase"]
        )

        winning_deltas["delta_diff_category"] = pd.cut(
            winning_deltas["delta_diff"],
            bins=[-float("inf"), q1, q3, float("inf")],
            labels=["decrease", "stable", "increase"]
        )

        for col in losing_deltas.select_dtypes(include="category").columns:
            losing_deltas[col] = losing_deltas[col].astype(str)

        for col in winning_deltas.select_dtypes(include="category").columns:
            winning_deltas[col] = winning_deltas[col].astype(str)

        # export data
        with open(f"../data/deltas_5_Losing Election_withcategories_{treatment}_{seed}.pkl", "wb") as f:
            pickle.dump(losing_deltas, f)

        with open(f"../data/deltas_5_Winning Election_withcategories_{treatment}_{seed}.pkl", "wb") as f:
            pickle.dump(winning_deltas, f)

else:
    # overall
    overall_deltas = pd.concat([losing_deltas, winning_deltas], ignore_index=True)

    # Get value of 25% and 75% quantiles
    q1 = overall_deltas["delta_diff"].quantile(0.25)
    q3 = overall_deltas["delta_diff"].quantile(0.75)

    losing_deltas["delta_diff_category"] = pd.cut(
        losing_deltas["delta_diff"],
        bins=[-float("inf"), q1, q3, float("inf")],
        labels=["decrease", "stable", "increase"]
    )

    winning_deltas["delta_diff_category"] = pd.cut(
        winning_deltas["delta_diff"],
        bins=[-float("inf"), q1, q3, float("inf")],
        labels=["decrease", "stable", "increase"]
    )

    for col in losing_deltas.select_dtypes(include="category").columns:
        losing_deltas[col] = losing_deltas[col].astype(str)

    for col in winning_deltas.select_dtypes(include="category").columns:
        winning_deltas[col] = winning_deltas[col].astype(str)

    # export data
    with open(f"../data/deltas_5_Losing Election_withcategories_{treatment}.pkl", "wb") as f:
        pickle.dump(losing_deltas, f)

    with open(f"../data/deltas_5_Winning Election_withcategories_{treatment}.pkl", "wb") as f:
        pickle.dump(winning_deltas, f)    

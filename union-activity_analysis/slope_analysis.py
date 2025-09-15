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

        # balance wins and losses per union

        losing_deltas["outcome"] = "Losing"
        winning_deltas["outcome"] = "Winning"

        deltas_all = pd.concat([losing_deltas, winning_deltas], ignore_index=True)

        # for each main union, balance the number of cases in losing and winning
        list_union_cases = []
        for union in deltas_all["main_union"].unique():
            # get losing and winning cases for this union
            losing_cases = deltas_all[(deltas_all["main_union"] == union) & (deltas_all["frame_prop"]=="rolling_community_prop") & (deltas_all["outcome"] == "Losing")]["case_number"].unique()
            winning_cases = deltas_all[(deltas_all["main_union"] == union) & (deltas_all["frame_prop"]=="rolling_community_prop") & (deltas_all["outcome"] == "Winning")]["case_number"].unique()

            cases_winning_sub = pd.Series(winning_cases).sample(n=len(losing_cases), random_state=42).tolist() if len(winning_cases) > len(losing_cases) else list(winning_cases)
            cases_losing_sub = pd.Series(losing_cases).sample(n=len(winning_cases), random_state=42).tolist() if len(losing_cases) > len(winning_cases) else list(losing_cases)

            list_union_cases += [(union, case, "Winning") for case in cases_winning_sub]
            list_union_cases += [(union, case, "Losing") for case in cases_losing_sub]

        deltas_all_sampled = deltas_all[deltas_all.apply(lambda row: (row["main_union"], row["case_number"], row["outcome"]) in list_union_cases, axis=1)].copy()
        losing_deltas_sampled = deltas_all_sampled[deltas_all_sampled["outcome"] == "Losing"]
        winning_deltas_sampled = deltas_all_sampled[deltas_all_sampled["outcome"] == "Winning"]

        losing_deltas, winning_deltas = subsample_unions(losing_deltas_sampled, winning_deltas_sampled, seed)

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
        with open(f"../data/deltas_5_Losing Election_withcategories_{treatment}_{seed}_sampled.pkl", "wb") as f:
            pickle.dump(losing_deltas, f)

        with open(f"../data/deltas_5_Winning Election_withcategories_{treatment}_{seed}_sampled.pkl", "wb") as f:
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

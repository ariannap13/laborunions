import bambi as bmb
import arviz as az
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys

seed = sys.argv[1]
rolling_window = sys.argv[2]

diag_prog_single = True
diag_or_prog = "diag"
balance_win_loss = True
subsample_overly_represented = False

if diag_prog_single:
    tag_diag_prog = "_diag+prog"
else:
    tag_diag_prog = "_"+diag_or_prog

if balance_win_loss:
    tag_balance_winloss = "_balanced"
else:
    tag_balance_winloss = ""

if subsample_overly_represented:
    tag_subsample_overly = "_subsample-active"
else:
    tag_subsample_overly = ""

data = pd.read_csv(f'../data/union_info_bayesian_rolling_window_{rolling_window}_seed{seed}{tag_diag_prog}{tag_balance_winloss}{tag_subsample_overly}-moredata.csv')

# remove unions whose min_count < 5
union_counts = data['union'].value_counts()
unions_to_keep = union_counts[union_counts >= 5].index
data = data[data['union'].isin(unions_to_keep)].reset_index(drop=True)

if diag_prog_single:
    X = data[[
        "diag+prog_prop",
        'mot_prop','comm_prop','eng_prop', 
        "anger_prob","disgust_prob","fear_prob","joy_prob","neutral_prob","sadness_prob","surprise_prob",
        "size", 
        "employees", 
        'assets', 
        'followers FB', 
        'Disbursement', 'Liabilities', 
        'weekly_posts_mean']].assign(const=1)
else:
    X = data[[
        "diag_prop",
        "prog_prop",
        'mot_prop','comm_prop','eng_prop', 
        "anger_prob","disgust_prob","fear_prob","joy_prob","neutral_prob","sadness_prob","surprise_prob",
        "size", 
        "employees", 
        'assets', 
        'followers FB', 
        'Disbursement', 'Liabilities', 
        'weekly_posts_mean']].assign(const=1)
    
vifs = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1]-1)],
                 index=X.columns[:-1])
print(vifs)

# rename columns
if diag_prog_single:
    data = data.rename(columns={"diag+prog_prop": "diagprog_prop"})
data = data.rename(columns={"followers FB": "followers_FB"})

if diag_prog_single:
    model = bmb.Model(
        "election_win ~ " \
        "diagprog_prop + " \
        "mot_prop + comm_prop + eng_prop + "\
        "anger_prob + disgust_prob + fear_prob + joy_prob + neutral_prob + sadness_prob + surprise_prob + "\
        "(1|union) + standardize(size) + "\
        "standardize(assets) + "\
        "standardize(Disbursement) + standardize(Liabilities) + standardize(weekly_posts_mean) +"\
        "primary_industry + type + rightwork + (1|election_state)",
        data=data,
        family="bernoulli",
    )
else:
    model = bmb.Model(
        "election_win ~ " \
        "diag_prop + " \
        "prog_prop + " \
        "mot_prop + comm_prop + eng_prop + "\
        "anger_prob + disgust_prob + fear_prob + joy_prob + neutral_prob + sadness_prob + surprise_prob + "\
        "(1|union) + standardize(size) + "\
        "standardize(assets) + "\
        "standardize(Disbursement) + standardize(Liabilities) + standardize(weekly_posts_mean) +"\
        "primary_industry + type + rightwork + (1|election_state)",
        data=data,
        family="bernoulli",
    )

# define and fit model
model_fitted = model.fit(
    draws=2000,
    tune=2000,
    target_accept=0.95,
    max_treedepth=15,
    cores=1
)

# save summary to csv
summary_df = az.summary(model_fitted, hdi_prob=0.90)
summary_df.to_csv(f"../data/bayesian_model_summary-rolling_window_{rolling_window}_seed{seed}{tag_diag_prog}{tag_balance_winloss}{tag_subsample_overly}-moredata.csv") # working file
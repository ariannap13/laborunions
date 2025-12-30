import bambi as bmb
import arviz as az
import pandas as pd
import formulae
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import sys

aggregation = "rolling"  # options: "rolling" or "all"
seed = sys.argv[1]
rolling_window = sys.argv[2]

data = pd.read_csv(f"../data/union_info_bayesian_{aggregation}_window_{rolling_window}_seed{seed}-moredata.csv")

# remove unions whose min_count < 5
union_counts = data['union'].value_counts()
unions_to_keep = union_counts[union_counts >= 5].index
data = data[data['union'].isin(unions_to_keep)].reset_index(drop=True)

# check how many rows have missing values
missing_values_count = data.isnull().sum()
print("Missing values in each column:")
print(missing_values_count)

print(data["primary_industry"].value_counts())

# evaluate collinearity
correlation_matrix = data[[
    #"diag+prog_prop",
    #"diag_prop", 
    "prog_prop", 
    "mot_prop", "comm_prop", "eng_prop", 
                           "anger_prob","disgust_prob","fear_prob","joy_prob","neutral_prob","sadness_prob","surprise_prob",
                           "size", 
                           #"employees", 
                           'assets', 'followers FB', 'Disbursement', 'Liabilities', 'total_posts_union', 'total_posts_before',
                           #"assets"
                           ]].corr()
# print(correlation_matrix)

X = data[[
    #"diag+prog_prop",
    #"diag_prop",
    "prog_prop",
    'mot_prop','comm_prop','eng_prop', 
          "anger_prob","disgust_prob","fear_prob","joy_prob","neutral_prob","sadness_prob","surprise_prob",
          "size", 
          #"employees", 
          #"assets"
          'assets', 
          #'followers FB', 
          'Disbursement', 'Liabilities', 
          #'total_posts_union', 
          'weekly_posts_mean',
          #'total_posts_before',
          ]].assign(const=1)
vifs = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1]-1)],
                 index=X.columns[:-1])
print(vifs)

# rename "diag+prog_prop" to "diagprog_prop" for formula compatibility
#data = data.rename(columns={"diag+prog_prop": "diagprog_prop"})
data = data.rename(columns={"followers FB": "followers_FB"})

model = bmb.Model(
    "election_win ~ " \
    "diagprog_prop + " \
    #"diag_prop + " \
    #"prog_prop + " \
    "mot_prop + comm_prop + eng_prop + "\
    "anger_prob + disgust_prob + fear_prob + joy_prob + neutral_prob + sadness_prob + surprise_prob + "\
    "(1|union) + standardize(size) + "\
    #"standardize(employees) + "\
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
summary_df.to_csv(f"../data/bayesian_model_summary-{aggregation}_prog_window_{rolling_window}_seed{seed}-moredata_staterandom.csv")
